import { LayerNode } from "@opencode-ai/core/effect/layer-node"
import { SessionV1 } from "@opencode-ai/core/v1/session"
import { ConfigV1 } from "@opencode-ai/core/v1/config/config"
import { Database } from "@opencode-ai/core/database/database"
import { NamedError } from "@opencode-ai/core/util/error"
import type { ModelMessage } from "ai"
import { Session } from "./session"
import { SessionID, MessageID, PartID } from "./schema"
import { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { MessageV2 } from "./message-v2"
import { Token } from "@/util/token"
import { SessionProcessor } from "./processor"
import type { LLM } from "./llm"
import { Agent } from "@/agent/agent"
import { Plugin } from "@/plugin"
import { Config } from "@/config/config"
import { NotFoundError } from "@/storage/storage"

import { Effect, Layer, Context } from "effect"
import { InstanceState } from "@/effect/instance-state"
import { isOverflow as overflow, usable } from "./overflow"
import { serviceUse } from "@opencode-ai/core/effect/service-use"
import { RuntimeFlags } from "@/effect/runtime-flags"
import { EventV2Bridge } from "@/event-v2-bridge"
import { ProviderV2 } from "@opencode-ai/core/provider"
import { ModelV2 } from "@opencode-ai/core/model"
import { buildPrompt } from "@opencode-ai/core/session/compaction"
import { SessionCompactionEvent } from "@opencode-ai/schema/session-compaction-event"

export const Event = SessionCompactionEvent

export const PRUNE_MINIMUM = 20_000
export const PRUNE_PROTECT = 40_000
const PRUNE_PROTECTED_TOOLS = ["skill"]
const MIN_PRESERVE_RECENT_TOKENS = 2_000
const MAX_PRESERVE_RECENT_TOKENS = 15_000
/** After each "too long" rejection without provider counts, the next request shrinks the refused size by this share. */
const SHRINK_STEPS = [0.85, 0.75, 0.65, 0.5]
/** The least of the window kept free for the summary, and the share of the window kept free when that is more. */
const RESERVE_MIN = 16_000
const RESERVE_SHARE = 0.15
/** A common window size, assumed for the summary request when the model's window is unknown. */
const UNKNOWN_WINDOW = 200_000
const OUTPUT_TOKEN_MIN = 1_024
/**
 * `Token.estimate` uses ~4 characters per token, while dense text and code on Claude/GPT tokenizers
 * often run ~2.8-2.9 characters per token (~1.38x).
 */
const ESTIMATE_ERROR = 0.4
const MAX_OBSERVED_DENSITY = 4
/**
 * How far one exchange of a summary transcript is shortened, in the order the levels apply: whole; tool outputs cut;
 * tool outputs left out and tool inputs cut; only the user's text, cut; omitted.
 */
const OUTPUTS_CLIPPED = 1
const OUTPUTS_REMOVED = 2
const USER_ONLY = 3
const OMITTED = 4
const TRANSCRIPT_LEVELS = [0, OUTPUTS_CLIPPED, OUTPUTS_REMOVED, USER_ONLY, OMITTED]
/** Characters kept, from the start and end together, of a cut tool input, tool output, or user message. */
const TRANSCRIPT_CLIP = 2_000

const SUMMARY_TEMPLATE = `You MUST use this format for your response (you may omit sections that aren't applicable). Do not include the <template> tags in your response.
<template>
## Objective
- [one or two brief sentences describing what the user is trying to accomplish]

## Requirements
- [constraints, preferences, requirements, and scope boundaries stated by the user, or "(none)"]

## Decisions
- [decisions already made and why, or "(none)"]

## Work State
Break the objective into smaller goals and report which are completed, which are being worked on, and which are blocked.
### Completed
- [goals that have been completed; otherwise "(none)"]

### Active
- [goals currently being worked on; otherwise "(none)"]

### Blocked
- [anything blocking progress, and why; otherwise "(none)"]

## Next Move
1. [ordered list of next actions, or "(none)"]

## Relevant Files
List the files and directories, other than the current working directory, that another agent would need to open to continue this work. Include at most 15, most important first. Do not list every file that was read or changed. Include paths outside the current working directory when relevant. If none, write "(none)".
- \`[file or directory path]\`: [brief reason it matters]

## Important Context
- [facts the next agent cannot continue without and cannot easily find on its own; or "(none)"]
</template>`

const SUMMARY_RULES = `Rules:
- Keep each section concise. Use terse, single-line bullets, not prose paragraphs or nested lists.
- Prefer short references over detailed restatement. It is fine to leave out information the next agent can recover from the code or the files listed above.
- Preserve exact file paths, symbols, commands, error strings, URLs, and identifiers.
- Carry forward only user questions or requests that remain unanswered or require further action. Do not repeat ones that newer history has answered or resolved. Preserve exact wording when carrying one forward.
- Preserve consequential workflow state, including whether changes are uncommitted, committed, pushed, under review, or merged.
- Do not mention the summary process or that context was compacted.`

const livePrompt = (update: boolean) => {
  const shared = [
    "Summarize only what the user and the assistant said and did. Leave out instructions and setup the assistant was given rather than told by the user: repository conventions, instruction files such as AGENTS.md, and environment details like the session ID. The next agent receives current versions of all of these separately.",
    SUMMARY_TEMPLATE,
    SUMMARY_RULES,
    "Do not continue the task or call tools.",
    "Return only the structured summary in the requested format. Do not include a preamble, explanation, or other commentary.",
  ]
  if (!update) {
    return [
      "You MUST summarize the conversation above into a structured summary that will be given to another agent to resume the work.",
      ...shared,
    ].join("\n\n")
  }
  return [
    "Update the existing checkpoint in the conversation above into one consolidated summary.",
    "Newer history always takes precedence over the existing checkpoint. Preserve previous information unless newer history clearly contradicts, supersedes, resolves, or makes it stale. If something is no longer relevant to continuing the work, you may remove it.",
    "Incorporate newer requirements, decisions, progress, and context. Reconcile Work State and Next Move: move completed work out of Active, remove resolved blockers and answered questions, and preserve unresolved or pending work.",
    ...shared,
  ].join("\n\n")
}

const TEXT_ONLY = "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools."
const NUDGE =
  "The previous response did not fill in the required summary template. Do not call tools. Return the summary as text using the exact section headings from the template."

const headings = (template: string) => template.split("\n").filter((line) => line.startsWith("##"))
const LIVE_HEADINGS = headings(SUMMARY_TEMPLATE)
const TEXT_HEADINGS = headings(buildPrompt({ context: [] }))
const filled = (text: string, expected: string[]) => text.split("\n").some((line) => expected.includes(line.trim()))

const withNudge = (messages: ModelMessage[]): ModelMessage[] => {
  const last = messages.at(-1)
  if (last?.role !== "user") return [...messages, { role: "user", content: NUDGE }]
  const content = typeof last.content === "string" ? [{ type: "text" as const, text: last.content }] : last.content
  return [...messages.slice(0, -1), { ...last, content: [...content, { type: "text", text: NUDGE }] }]
}

/** `value` cut to `TRANSCRIPT_CLIP` characters from its start and end, never splitting a surrogate pair. */
const clip = (value: string) => {
  if (value.length <= TRANSCRIPT_CLIP) return value
  const chars = Array.from(value)
  if (chars.length <= TRANSCRIPT_CLIP) return value
  const half = TRANSCRIPT_CLIP / 2
  return `${chars.slice(0, half).join("")}\n[... ${chars.length - TRANSCRIPT_CLIP} characters omitted ...]\n${chars.slice(-half).join("")}`
}

/** One stored message as transcript lines at a shortening level; empty for what that level leaves out. */
const render = (message: SessionV1.WithParts, level: number) => {
  if (level === OMITTED) return ""
  if (message.info.role === "user") {
    const text = message.parts
      .filter((part): part is SessionV1.TextPart => part.type === "text" && !part.ignored)
      .map((part) => part.text)
      .filter(Boolean)
      .join("\n")
    const said = text ? [`[User]: ${level >= USER_ONLY ? clip(text) : text}`] : []
    if (level >= USER_ONLY) return said.join("\n")
    const files = message.parts.flatMap((part) =>
      part.type === "file" ? [`[Attached ${part.mime}: ${part.filename ?? "file"}]`] : [],
    )
    return [...said, ...files].join("\n")
  }
  if (level >= USER_ONLY) return ""
  // The model's reasoning stays out: the provider drops it from the conversation's own requests,
  // so it would make the summary request larger than the context it summarizes.
  return message.parts
    .flatMap((part) => {
      if (part.type === "text") return part.text ? [`[Assistant]: ${part.text}`] : []
      if (part.type !== "tool") return []
      const input = JSON.stringify(part.state.input) ?? ""
      const call = `[Assistant tool call]: ${part.tool}(${level >= OUTPUTS_REMOVED ? clip(input) : input})`
      if (part.state.status === "completed") {
        if (level >= OUTPUTS_REMOVED) return [call, "[Tool result omitted]"]
        const attachments = (part.state.attachments ?? []).map(
          (item) => `[Attached ${item.mime}: ${item.filename ?? "file"}]`,
        )
        const output = part.state.time.compacted
          ? "[Old tool result content cleared]"
          : [part.state.output, ...attachments].join("\n")
        return [call, `[Tool result]: ${level >= OUTPUTS_CLIPPED ? clip(output) : output}`]
      }
      if (part.state.status === "error") {
        if (level >= OUTPUTS_REMOVED) return [call, "[Tool error omitted]"]
        return [call, `[Tool error]: ${level >= OUTPUTS_CLIPPED ? clip(part.state.error) : part.state.error}`]
      }
      return [call]
    })
    .join("\n")
}

const exchanges = (messages: SessionV1.WithParts[]) =>
  messages
    .reduce<SessionV1.WithParts[][]>((groups, message) => {
      if (message.info.role === "user" || groups.length === 0) groups.push([message])
      else groups[groups.length - 1].push(message)
      return groups
    }, [])
    .map((group) =>
      TRANSCRIPT_LEVELS.map((level) =>
        group
          .map((message) => render(message, level))
          .filter(Boolean)
          .join("\n"),
      ),
    )
    .filter((renders) => renders[0])

const note = (omitted: number) =>
  omitted ? `[${omitted} older ${omitted === 1 ? "exchange" : "exchanges"} omitted]` : ""
const size = (text: string) => (text ? Token.estimate(text) + 1 : 0)

/**
 * The transcript within `room` tokens. Exchanges are shortened oldest first, one level at a time, so detail goes from
 * the oldest history first and whole exchanges only as a last resort; the newest is never omitted. `keepFirst` holds
 * the opening exchange, which states the task, at its shortest until every other exchange that can go has gone.
 * Undefined when the transcript cannot fit even so.
 */
const fit = (renders: string[][], room: number, keepFirst: boolean) => {
  // Room for the longest possible note is set aside before choosing what to keep.
  const budget = room - Token.estimate(note(renders.length))
  const levels = renders.map(() => 0)
  let total = renders.reduce((sum, item) => sum + size(item[0]), 0)
  const newest = renders.length - 1
  for (const level of TRANSCRIPT_LEVELS.slice(1)) {
    for (let index = 0; index < renders.length && total > budget; index++) {
      if (index === newest && !renders[index][level]) continue
      if (keepFirst && index === 0 && level === OMITTED) continue
      total += size(renders[index][level]) - size(renders[index][levels[index]])
      levels[index] = level
    }
  }
  if (total > budget && keepFirst && newest > 0) {
    total -= size(renders[0][levels[0]])
    levels[0] = OMITTED
  }
  if (total > budget) return undefined
  const kept = renders.map((item, index) => item[levels[index]])
  const first = kept.findIndex((text) => !text)
  if (first < 0) return kept.join("\n\n")
  return [...kept.slice(0, first), note(kept.filter((text) => !text).length), ...kept.slice(first)]
    .filter(Boolean)
    .join("\n\n")
}

/**
 * The largest prompt the model takes, or 0 when unknown. The context window also holds the reply, so an input limit
 * above it cannot be used in full.
 */
const inputWindow = (limit: Provider.Model["limit"]) =>
  limit.input && limit.context ? Math.min(limit.input, limit.context) : limit.input || limit.context

/** The largest summary request that still leaves the window's reserve for the summary. */
const ceiling = (window: number) =>
  window - Math.max(Math.floor(window * RESERVE_SHARE), window >= 2 * RESERVE_MIN ? RESERVE_MIN : 0)

/** The largest unchanged request that leaves `RESERVE_MIN` for the summary, or less when it cannot generate that much. */
const summaryRoom = (limit: Provider.Model["limit"], output: number) =>
  Math.max(0, inputWindow(limit) - Math.min(RESERVE_MIN, output))

/**
 * The output limit for a summary request of `input` tokens: the model's own, lowered to the room the request leaves
 * in the effective context window after scaling estimated tokens by `ESTIMATE_ERROR` (or observed provider density).
 * Thinking counts against the limit, so a fixed cap below that room can end a summary before it has written a word.
 */
const fitOutput = (
  model: Provider.Model,
  outputTokenMax: number | undefined,
  input: { measured: number; estimated: number; context?: number; scale?: number },
) => {
  const requested = ProviderTransform.maxOutputTokens(model, outputTokenMax)
  const context = input.context ?? model.limit.context
  if (context <= 0) return requested
  const scale = Math.max(1 + ESTIMATE_ERROR, input.scale ?? 1 + ESTIMATE_ERROR)
  const room = Math.max(OUTPUT_TOKEN_MIN, context - input.measured - Math.ceil(input.estimated * scale))
  return Math.min(requested, room)
}

type OverflowLimit = {
  actual?: number
  max: number
}

const parseCount = (raw: string) => Number(raw.replaceAll(",", ""))

/** Extracts provider-reported actual and/or maximum token counts from a context-overflow error. */
const parseOverflowLimit = (error: SessionV1.Assistant["error"]): OverflowLimit | undefined => {
  if (!SessionV1.ContextOverflowError.isInstance(error)) return undefined
  const text = [error.data.message, error.data.responseBody].filter(Boolean).join("\n")
  if (!text) return undefined

  // Anthropic: "input length and `max_tokens` exceed context limit: 920000 + 128000 > 1000000"
  const sum = /exceed context limit:\s*([\d,]+)\s*\+\s*([\d,]+)\s*>\s*([\d,]+)/i.exec(text)
  if (sum) {
    const actual = parseCount(sum[1])
    const max = parseCount(sum[3])
    if (actual > 0 && max > 0) return { actual, max }
  }

  // Anthropic: "prompt is too long: 491235 tokens > 135000 maximum"
  const gt = /([\d,]+)\s*tokens?\s*>\s*([\d,]+)\s*maximum/i.exec(text)
  if (gt) {
    const actual = parseCount(gt[1])
    const max = parseCount(gt[2])
    if (actual > 0 && max > 0) return { actual, max }
  }

  // OpenAI / vLLM: "maximum context length is 128000 tokens. However, ... resulted in / requested 145000 tokens (130000 in the messages, ...)"
  const oai =
    /maximum context length is\s*([\d,]+)\s*tokens[\s\S]*?(?:resulted in|requested)\s*([\d,]+)\s*tokens(?:\s*\(\s*([\d,]+)\s*in the messages)?/i.exec(
      text,
    )
  if (oai) {
    const max = parseCount(oai[1])
    const actual = parseCount(oai[3] ?? oai[2])
    if (actual > 0 && max > 0) return { actual, max }
  }

  // Gemini / Vertex: "input token count (150000) exceeds the maximum ... (128000)"
  const gemini = /input token count\s*\(([\d,]+)\)[\s\S]*?maximum[^(\d]*\(?([\d,]+)\)?/i.exec(text)
  if (gemini) {
    const actual = parseCount(gemini[1])
    const max = parseCount(gemini[2])
    if (actual > 0 && max > 0) return { actual, max }
  }

  // Llama.cpp / Ollama / generic pair formats
  const pair =
    /(?:input\s*\(([\d,]+)\s*tokens?\)\s*is longer than[^(\d]*\(([\d,]+)\s*tokens?\)|prompt has\s*([\d,]+)\s*tokens?,\s*but the configured context size is\s*([\d,]+)\s*tokens?)/i.exec(
      text,
    )
  if (pair) {
    const actual = parseCount(pair[1] ?? pair[3])
    const max = parseCount(pair[2] ?? pair[4])
    if (actual > 0 && max > 0) return { actual, max }
  }

  const maxOnly =
    /(?:maximum context length(?: of|\s+is)?\s*\(?([\d,]+)|maximum prompt length is\s*([\d,]+)|context length is only\s*([\d,]+)|with\s*([\d,]+)\s*maximum context length|exceeds the limit of\s*([\d,]+))/i.exec(
      text,
    )
  if (maxOnly) {
    const max = parseCount(maxOnly[1] ?? maxOnly[2] ?? maxOnly[3] ?? maxOnly[4] ?? maxOnly[5])
    if (max > 0) return { max }
  }

  return undefined
}

type Turn = {
  start: number
  end: number
  id: MessageID
}

type Tail = {
  start: number
  id: MessageID
}

type CompletedCompaction = {
  userIndex: number
  assistantIndex: number
  summary: string | undefined
}

function summaryText(message: SessionV1.WithParts) {
  const text = message.parts
    .filter((part): part is SessionV1.TextPart => part.type === "text")
    .map((part) => part.text.trim())
    .filter(Boolean)
    .join("\n\n")
    .trim()
  return text || undefined
}

function completedCompactions(messages: SessionV1.WithParts[]) {
  const users = new Map<MessageID, number>()
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.info.role !== "user") continue
    if (!msg.parts.some((part) => part.type === "compaction")) continue
    users.set(msg.info.id, i)
  }

  return messages.flatMap((msg, assistantIndex): CompletedCompaction[] => {
    if (msg.info.role !== "assistant") return []
    if (!msg.info.summary || !msg.info.finish || msg.info.error) return []
    const userIndex = users.get(msg.info.parentID)
    if (userIndex === undefined) return []
    return [{ userIndex, assistantIndex, summary: summaryText(msg) }]
  })
}

function preserveRecentBudget(input: { cfg: ConfigV1.Info; model: Provider.Model }) {
  return (
    input.cfg.compaction?.preserve_recent_tokens ??
    Math.min(MAX_PRESERVE_RECENT_TOKENS, Math.max(MIN_PRESERVE_RECENT_TOKENS, Math.floor(usable(input) * 0.25)))
  )
}

function turns(messages: SessionV1.WithParts[]) {
  const result: Turn[] = []
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.info.role !== "user") continue
    if (msg.parts.some((part) => part.type === "compaction")) continue
    result.push({
      start: i,
      end: messages.length,
      id: msg.info.id,
    })
  }
  for (let i = 0; i < result.length - 1; i++) {
    result[i].end = result[i + 1].start
  }
  return result
}

function splitTurn(input: {
  messages: SessionV1.WithParts[]
  turn: Turn
  model: Provider.Model
  budget: number
  estimate: (input: { messages: SessionV1.WithParts[]; model: Provider.Model }) => Effect.Effect<number>
}) {
  return Effect.gen(function* () {
    if (input.budget <= 0) return undefined
    if (input.turn.end - input.turn.start <= 1) return undefined
    for (let start = input.turn.start + 1; start < input.turn.end; start++) {
      const size = yield* input.estimate({
        messages: input.messages.slice(start, input.turn.end),
        model: input.model,
      })
      if (size > input.budget) continue
      return {
        start,
        id: input.messages[start]!.info.id,
      } satisfies Tail
    }
    return undefined
  })
}

/**
 * Builds the request the session's next turn would send for `messages`, with `prompt` appended as its last message.
 * Undefined when it cannot.
 */
export type Live = (input: {
  messages: SessionV1.WithParts[]
  user: SessionV1.User
  model: Provider.Model
  processor: SessionProcessor.Handle
  prompt: SessionV1.WithParts
}) => Effect.Effect<LLM.StreamInput | undefined>

export interface Interface {
  readonly isOverflow: (input: {
    tokens: SessionV1.Assistant["tokens"]
    model: Provider.Model
  }) => Effect.Effect<boolean>
  readonly prune: (input: { sessionID: SessionID }) => Effect.Effect<void>
  readonly process: (input: {
    parentID: MessageID
    messages: SessionV1.WithParts[]
    sessionID: SessionID
    auto: boolean
    overflow?: boolean
    live?: Live
  }) => Effect.Effect<"continue" | "stop">
  readonly create: (input: {
    sessionID: SessionID
    agent: string
    model: { providerID: ProviderV2.ID; modelID: ModelV2.ID }
    auto: boolean
    overflow?: boolean
  }) => Effect.Effect<void>
}

export class Service extends Context.Service<Service, Interface>()("@opencode/SessionCompaction") {}

export const use = serviceUse(Service)

const layer = Layer.effect(
  Service,
  Effect.gen(function* () {
    const config = yield* Config.Service
    const session = yield* Session.Service
    const agents = yield* Agent.Service
    const plugin = yield* Plugin.Service
    const processors = yield* SessionProcessor.Service
    const provider = yield* Provider.Service
    const events = yield* EventV2Bridge.Service
    const flags = yield* RuntimeFlags.Service
    const database = yield* Database.Service

    const isOverflow = Effect.fn("SessionCompaction.isOverflow")(function* (input: {
      tokens: SessionV1.Assistant["tokens"]
      model: Provider.Model
    }) {
      return overflow({
        cfg: yield* config.get(),
        tokens: input.tokens,
        model: input.model,
        outputTokenMax: flags.outputTokenMax,
      })
    })

    const estimate = Effect.fn("SessionCompaction.estimate")(function* (input: {
      messages: SessionV1.WithParts[]
      model: Provider.Model
    }) {
      // Providers drop old reasoning from the requests they are sent, so it does not count.
      const msgs = yield* MessageV2.toModelMessagesEffect(
        input.messages.map((message) => ({
          ...message,
          parts: message.parts.filter((part) => part.type !== "reasoning"),
        })),
        input.model,
      )
      return Token.estimate(JSON.stringify(msgs))
    })

    const select = Effect.fn("SessionCompaction.select")(function* (input: {
      messages: SessionV1.WithParts[]
      cfg: ConfigV1.Info
      model: Provider.Model
    }) {
      const limit = input.cfg.compaction?.tail_turns
      if (limit !== undefined && limit <= 0) return { head: input.messages, tail_start_id: undefined }
      const budget = preserveRecentBudget({ cfg: input.cfg, model: input.model })
      const all = turns(input.messages)
      if (!all.length) return { head: input.messages, tail_start_id: undefined }
      const recent = limit === undefined ? all : all.slice(-limit)

      let total = 0
      let keep: Tail | undefined
      for (let i = recent.length - 1; i >= 0; i--) {
        const turn = recent[i]!
        // estimate lazily so cost stays proportional to the retained tail, not the whole session
        const size = yield* estimate({
          messages: input.messages.slice(turn.start, turn.end),
          model: input.model,
        })
        if (total + size <= budget) {
          total += size
          keep = { start: turn.start, id: turn.id }
          continue
        }
        const remaining = budget - total
        const split = yield* splitTurn({
          messages: input.messages,
          turn,
          model: input.model,
          budget: remaining,
          estimate,
        })
        if (split) keep = split
        else if (!keep) {
          yield* Effect.logInfo("tail fallback", { budget, size, total })
        }
        break
      }

      if (!keep || keep.start === 0) return { head: input.messages, tail_start_id: undefined }
      return {
        head: input.messages.slice(0, keep.start),
        tail_start_id: keep.id,
      }
    })

    // goes backwards through parts until there are PRUNE_PROTECT tokens worth of tool
    // calls, then erases output of older tool calls to free context space
    const prune = Effect.fn("SessionCompaction.prune")(function* (input: { sessionID: SessionID }) {
      const cfg = yield* config.get()
      if (!cfg.compaction?.prune) return
      yield* Effect.logInfo("pruning")

      const msgs = yield* session
        .messages({ sessionID: input.sessionID })
        .pipe(Effect.catchIf(NotFoundError.isInstance, () => Effect.succeed(undefined)))
      if (!msgs) return

      let total = 0
      let pruned = 0
      const toPrune: SessionV1.ToolPart[] = []
      let turns = 0

      loop: for (let msgIndex = msgs.length - 1; msgIndex >= 0; msgIndex--) {
        const msg = msgs[msgIndex]
        if (msg.info.role === "user") turns++
        if (turns < 2) continue
        if (msg.info.role === "assistant" && msg.info.summary) break loop
        for (let partIndex = msg.parts.length - 1; partIndex >= 0; partIndex--) {
          const part = msg.parts[partIndex]
          if (part.type !== "tool") continue
          if (part.state.status !== "completed") continue
          if (PRUNE_PROTECTED_TOOLS.includes(part.tool)) continue
          if (part.state.time.compacted) break loop
          const estimate = Token.estimate(part.state.output)
          total += estimate
          if (total <= PRUNE_PROTECT) continue
          pruned += estimate
          toPrune.push(part)
        }
      }

      yield* Effect.logInfo("found", { pruned, total })
      if (pruned > PRUNE_MINIMUM) {
        for (const part of toPrune) {
          if (part.state.status === "completed") {
            part.state.time.compacted = Date.now()
            yield* session.updatePart(part)
          }
        }
        yield* Effect.logInfo("pruned", { count: toPrune.length })
      }
    })

    const processCompaction = Effect.fn("SessionCompaction.process")(function* (input: {
      parentID: MessageID
      messages: SessionV1.WithParts[]
      sessionID: SessionID
      auto: boolean
      overflow?: boolean
      live?: Live
    }) {
      const parent = input.messages.findLast((m) => m.info.id === input.parentID)
      if (!parent || parent.info.role !== "user") {
        throw new Error(`Compaction parent must be a user message: ${input.parentID}`)
      }
      const userMessage = parent.info
      const compactionPart = parent.parts.find((part): part is SessionV1.CompactionPart => part.type === "compaction")

      let messages = input.messages
      let replay:
        | {
            info: SessionV1.User
            parts: SessionV1.Part[]
          }
        | undefined
      if (input.overflow) {
        const idx = input.messages.findIndex((m) => m.info.id === input.parentID)
        for (let i = idx - 1; i >= 0; i--) {
          const msg = input.messages[i]
          if (msg.info.role === "user" && !msg.parts.some((p) => p.type === "compaction")) {
            replay = { info: msg.info, parts: msg.parts }
            messages = input.messages.slice(0, i)
            break
          }
        }
        const hasContent =
          replay && messages.some((m) => m.info.role === "user" && !m.parts.some((p) => p.type === "compaction"))
        if (!hasContent) {
          replay = undefined
          messages = input.messages
        }
      }

      const agent = yield* agents.get("compaction")
      const sessionModel = yield* provider
        .getModel(userMessage.model.providerID, userMessage.model.modelID)
        .pipe(Effect.orDie)
      const model = agent.model
        ? yield* provider.getModel(agent.model.providerID, agent.model.modelID).pipe(Effect.orDie)
        : sessionModel
      const cfg = yield* config.get()
      const history = compactionPart && messages.at(-1)?.info.id === input.parentID ? messages.slice(0, -1) : messages
      const prior = completedCompactions(history)
      const hidden = new Set(prior.flatMap((item) => [item.userIndex, item.assistantIndex]))
      const previousSummary = prior.at(-1)?.summary
      const selected = yield* select({
        messages: history.filter((_, index) => !hidden.has(index)),
        cfg,
        // The retained tail is sent to the session's model, so it is sized for that model.
        model: sessionModel,
      })
      // Allow plugins to inject context or replace compaction prompt.
      const compacting = yield* plugin.trigger(
        "experimental.session.compacting",
        { sessionID: input.sessionID },
        { context: [], prompt: undefined },
      )
      const ctx = yield* InstanceState.context
      const msg: SessionV1.Assistant = {
        id: MessageID.ascending(),
        role: "assistant",
        parentID: input.parentID,
        sessionID: input.sessionID,
        mode: "compaction",
        agent: "compaction",
        variant: userMessage.model.variant,
        summary: true,
        path: {
          cwd: ctx.directory,
          root: ctx.worktree,
        },
        cost: 0,
        tokens: {
          output: 0,
          input: 0,
          reasoning: 0,
          cache: { read: 0, write: 0 },
        },
        modelID: model.id,
        providerID: model.providerID,
        time: {
          created: Date.now(),
        },
      }
      yield* session.updateMessage(msg)
      const processor = yield* processors.create({
        assistantMessage: msg,
        sessionID: input.sessionID,
        model,
      })
      const parts = () => MessageV2.parts(msg.id).pipe(Effect.provideService(Database.Service, database))
      const reply = Effect.fnUntraced(function* () {
        return summaryText({ info: msg, parts: yield* parts() }) ?? ""
      })
      // Every attempt writes into the same message, so each starts from no parts, finish, or error.
      const reset = Effect.fnUntraced(function* () {
        for (const part of yield* parts()) {
          yield* session.removePart({ sessionID: msg.sessionID, messageID: msg.id, partID: part.id })
        }
        msg.finish = undefined
        msg.error = undefined
        msg.time.completed = undefined
        yield* session.updateMessage(msg)
      })
      // A reply that ran out of output tokens is not a summary: thinking may have used them all, or the text stops short.
      const usable = (result: SessionProcessor.Result) =>
        result === "continue" && !msg.error && msg.finish !== "tool-calls" && msg.finish !== "length"
      let effectiveContext = model.limit.context
      let effectiveWindow = inputWindow(model.limit) || UNKNOWN_WINDOW
      let observedScale = 1 + ESTIMATE_ERROR
      const learnOverflow = (error: SessionV1.Assistant["error"], estimated?: number) => {
        const limit = parseOverflowLimit(error)
        if (!limit) return undefined
        effectiveContext = effectiveContext > 0 ? Math.min(effectiveContext, limit.max) : limit.max
        effectiveWindow = Math.min(effectiveWindow, limit.max)
        if (limit.actual && estimated && estimated > 0) {
          observedScale = Math.max(observedScale, Math.min(MAX_OBSERVED_DENSITY, (limit.actual / estimated) * 1.05))
        }
        return limit
      }
      const output = (size: { measured: number; estimated: number }) =>
        fitOutput(model, flags.outputTokenMax, { ...size, context: effectiveContext, scale: observedScale })

      /**
       * The conversation exactly as the session's next turn would send it, then the summary prompt, so the unchanged
       * prefix reuses that turn's prompt cache. Only while that request is the one the latest response measured and
       * still leaves the summary room; "fallback", with the attempt cleared, when it is skipped or its reply is
       * unusable; "error" when explicitly aborted.
       */
      const summarizeLive = Effect.fnUntraced(function* () {
        if (!input.live || input.overflow) return "fallback" as const
        const user = history.findLast(
          (item) => item.info.role === "user" && !item.parts.some((part) => part.type === "compaction"),
        )?.info
        if (user?.role !== "user" || user.format?.type === "json_schema") return "fallback" as const
        if (user.model.providerID !== model.providerID || user.model.modelID !== model.id) return "fallback" as const
        const anchorIndex = history.findLastIndex(
          (item) =>
            item.info.role === "assistant" &&
            !item.info.summary &&
            item.info.finish !== undefined &&
            !item.info.error,
        )
        const anchor = history[anchorIndex]
        if (
          anchor?.info.role !== "assistant" ||
          anchor.info.providerID !== model.providerID ||
          anchor.info.modelID !== model.id ||
          anchor.info.variant !== user.model.variant ||
          anchor.info.agent !== user.agent
        )
          return "fallback" as const
        const tokens = anchor.info.tokens
        const prompted = tokens.input + tokens.cache.read + tokens.cache.write
        if (prompted === 0) return "fallback" as const
        const prompt = [
          compacting.prompt ?? livePrompt(previousSummary !== undefined),
          TEXT_ONLY,
          ...(compacting.prompt === undefined ? compacting.context : []),
        ].join("\n\n")
        // The response measured everything before it, but not its own tool results or anything since.
        const results = anchor.parts.reduce((sum, part) => {
          if (part.type !== "tool") return sum
          if (part.state.status === "completed") return sum + Token.estimate(part.state.output)
          if (part.state.status === "error") return sum + Token.estimate(part.state.error)
          return sum
        }, 0)
        const size = {
          measured: prompted + tokens.output + tokens.reasoning,
          estimated:
            (yield* estimate({ messages: history.slice(anchorIndex + 1), model })) + results + Token.estimate(prompt),
        }
        const room = Math.max(
          ceiling(effectiveWindow),
          summaryRoom(model.limit, ProviderTransform.maxOutputTokens(model, flags.outputTokenMax)),
        )
        if (size.measured + Math.ceil((size.estimated + Token.estimate(NUDGE)) * observedScale) > room)
          return "fallback" as const

        const id = MessageID.ascending()
        const request = yield* input.live({
          messages: structuredClone(history),
          user,
          model,
          processor,
          prompt: {
            info: {
              id,
              role: "user",
              sessionID: input.sessionID,
              agent: user.agent,
              model: user.model,
              time: { created: Date.now() },
            },
            parts: [
              { id: PartID.ascending(), messageID: id, sessionID: input.sessionID, type: "text", text: prompt, synthetic: true },
            ],
          },
        })
        if (!request) return "fallback" as const
        // The tools stay defined so the prefix still matches, but one the summary calls must not run.
        const tools = Object.fromEntries(
          Object.entries(request.tools).map(([name, tool]) => [name, { ...tool, execute: undefined }]),
        )
        const send = (messages: ModelMessage[], extra: number) =>
          processor.process({
            ...request,
            tools,
            messages,
            maxOutputTokens: output({ measured: size.measured, estimated: size.estimated + extra }),
          })
        const result = yield* send(request.messages, 0)
        if (result === "compact") learnOverflow(msg.error)
        if (SessionV1.AbortedError.isInstance(msg.error)) return "error" as const
        const text = yield* reply()
        if (usable(result) && text) {
          if (compacting.prompt !== undefined || filled(text, LIVE_HEADINGS)) return "summarized" as const
          yield* reset()
          const nudged = yield* send(withNudge(request.messages), Token.estimate(NUDGE))
          if (nudged === "compact") learnOverflow(msg.error)
          if (SessionV1.AbortedError.isInstance(msg.error)) return "error" as const
          if (usable(nudged) && filled(yield* reply(), LIVE_HEADINGS)) return "summarized" as const
        }
        yield* reset()
        return "fallback" as const
      })

      /**
       * The history before the retained tail as one text transcript, without tools. It is shortened to fit the window,
       * and again—calibrated from provider token counts when reported, or shrinking the refused size by `SHRINK_STEPS`
       * otherwise—each time the model refuses one as too long.
       */
      const summarizeText = Effect.fnUntraced(function* () {
        const msgs = structuredClone(selected.head)
        yield* plugin.trigger("experimental.chat.messages.transform", {}, { messages: msgs })
        const renders = exchanges(msgs)
        const promptFor = (conversation: string) => {
          const nextPrompt =
            compacting.prompt ??
            [buildPrompt({ previousSummary, context: [conversation] }), ...compacting.context]
              .filter(Boolean)
              .join("\n\n")
          return [nextPrompt, ...(compacting.prompt ? ["The following is the conversation history:", conversation] : [])]
            .filter(Boolean)
            .join("\n\n")
        }
        const system = Token.estimate(agent.prompt ?? "")
        const fixed = Token.estimate(promptFor("")) + system + Token.estimate(NUDGE)
        let target = ceiling(effectiveWindow)
        let refused: string | undefined
        for (let rejections = 0; ; ) {
          // Until the model refuses one as too long, a transcript that cannot be shortened enough is sent whole.
          const conversation =
            fit(renders, target - fixed, previousSummary === undefined) ??
            (rejections === 0 ? fit(renders, Number.POSITIVE_INFINITY, false) : undefined)
          if (conversation === undefined) return "overflow" as const
          // The shortening goes level by level, so a smaller target can leave the refused transcript as it was.
          if (conversation === refused) {
            if (rejections === SHRINK_STEPS.length) return "overflow" as const
            rejections++
            target = Math.floor(target * SHRINK_STEPS[rejections - 1])
            continue
          }
          const text = promptFor(conversation)
          const estimated = Token.estimate(text) + system
          const messages: ModelMessage[] = [{ role: "user", content: [{ type: "text", text }] }]
          const send = (messages: ModelMessage[], extra: number) =>
            processor.process({
              user: userMessage,
              agent,
              sessionID: input.sessionID,
              tools: {},
              system: [],
              messages,
              model,
              maxOutputTokens: output({ measured: 0, estimated: estimated + extra }),
            })
          const result = yield* send(messages, 0)
          if (result === "compact") {
            if (rejections === SHRINK_STEPS.length) return "overflow" as const
            const limit = learnOverflow(msg.error, estimated)
            refused = conversation
            rejections++
            const blind = Math.floor(estimated * SHRINK_STEPS[rejections - 1])
            const calibrated =
              limit?.actual !== undefined
                ? Math.floor(estimated * (ceiling(limit.max) / limit.actual) * 0.92)
                : limit !== undefined
                  ? Math.floor(ceiling(limit.max) / observedScale)
                  : undefined
            target = calibrated !== undefined ? Math.min(blind, calibrated) : blind
            yield* reset()
            continue
          }
          if (msg.finish === "tool-calls") return "tool-calls" as const
          if (!msg.error && msg.finish === "length") return "cut-off" as const
          if (!usable(result)) return "error" as const
          const first = yield* reply()
          // Kept as a summary, either would stand in for the whole history; failed, the next turn asks again.
          if (!first) return "empty" as const
          if (compacting.prompt !== undefined || filled(first, TEXT_HEADINGS)) return "summarized" as const
          // One reminder; if it does not help, the first summary is kept rather than none at all.
          const kept = { parts: yield* parts(), finish: msg.finish, tokens: msg.tokens }
          yield* reset()
          const nudged = yield* send(withNudge(messages), Token.estimate(NUDGE))
          if (usable(nudged) && filled(yield* reply(), TEXT_HEADINGS)) return "summarized" as const
          yield* reset()
          for (const part of kept.parts) yield* session.updatePart(part)
          msg.finish = kept.finish
          msg.tokens = kept.tokens
          msg.time.completed = Date.now()
          yield* session.updateMessage(msg)
          return "summarized" as const
        }
      })

      const liveOutcome = yield* summarizeLive()
      const outcome = liveOutcome === "fallback" ? yield* summarizeText() : liveOutcome
      if (outcome !== "summarized") {
        if (outcome === "overflow") {
          msg.error = new SessionV1.ContextOverflowError({
            message: replay
              ? "Conversation history too large to compact - exceeds model context limit"
              : "Session too large to compact - context exceeds model limit even after stripping media",
          }).toObject()
        }
        if (outcome === "tool-calls") {
          msg.error = new NamedError.Unknown({
            message: "Compaction failed: the model called a tool instead of writing the summary",
          }).toObject()
        }
        if (outcome === "cut-off") {
          msg.error = new NamedError.Unknown({
            message: "Compaction failed: the summary ran out of output tokens",
          }).toObject()
        }
        if (outcome === "empty") {
          msg.error = new NamedError.Unknown({ message: "Compaction failed: the model wrote no summary" }).toObject()
        }
        if (outcome !== "error") msg.finish = "error"
        msg.time.completed ??= Date.now()
        yield* session.updateMessage(msg)
        if (msg.error) yield* events.publish(Session.Event.Error, { sessionID: input.sessionID, error: msg.error })
        return "stop" as const
      }

      if (compactionPart && selected.tail_start_id && compactionPart.tail_start_id !== selected.tail_start_id) {
        yield* session.updatePart({
          ...compactionPart,
          tail_start_id: selected.tail_start_id,
        })
      }

      if (input.auto) {
        if (replay) {
          const original = replay.info
          const replayMsg = yield* session.updateMessage({
            id: MessageID.ascending(),
            role: "user",
            sessionID: input.sessionID,
            time: { created: Date.now() },
            agent: original.agent,
            model: original.model,
            format: original.format,
            tools: original.tools,
            system: original.system,
          })
          for (const part of replay.parts) {
            if (part.type === "compaction") continue
            const replayPart =
              part.type === "file" && MessageV2.isMedia(part.mime)
                ? { type: "text" as const, text: `[Attached ${part.mime}: ${part.filename ?? "file"}]` }
                : part
            yield* session.updatePart({
              ...replayPart,
              id: PartID.ascending(),
              messageID: replayMsg.id,
              sessionID: input.sessionID,
            })
          }
        }

        if (!replay) {
          const info = yield* provider.getProvider(userMessage.model.providerID)
          if (
            (yield* plugin.trigger(
              "experimental.compaction.autocontinue",
              {
                sessionID: input.sessionID,
                agent: userMessage.agent,
                model: yield* provider
                  .getModel(userMessage.model.providerID, userMessage.model.modelID)
                  .pipe(Effect.orDie),
                provider: {
                  source: info.source,
                  info,
                  options: info.options,
                },
                message: userMessage,
                overflow: input.overflow === true,
              },
              { enabled: true },
            )).enabled
          ) {
            const continueMsg = yield* session.updateMessage({
              id: MessageID.ascending(),
              role: "user",
              sessionID: input.sessionID,
              time: { created: Date.now() },
              agent: userMessage.agent,
              model: userMessage.model,
            })
            const text =
              (input.overflow
                ? "The previous request exceeded the provider's size limit due to large media attachments. The conversation was compacted and media files were removed from context. If the user was asking about attached images or files, explain that the attachments were too large to process and suggest they try again with smaller or fewer files.\n\n"
                : "") +
              "Continue if you have next steps, or stop and ask for clarification if you are unsure how to proceed."
            yield* session.updatePart({
              id: PartID.ascending(),
              messageID: continueMsg.id,
              sessionID: input.sessionID,
              type: "text",
              // Internal marker for auto-compaction followups so provider plugins
              // can distinguish them from manual post-compaction user prompts.
              // This is not a stable plugin contract and may change or disappear.
              metadata: { compaction_continue: true },
              synthetic: true,
              text,
              time: {
                start: Date.now(),
                end: Date.now(),
              },
            })
          }
        }
      }

      yield* events.publish(Event.Compacted, { sessionID: input.sessionID })
      return "continue" as const
    })

    const create = Effect.fn("SessionCompaction.create")(function* (input: {
      sessionID: SessionID
      agent: string
      model: { providerID: ProviderV2.ID; modelID: ModelV2.ID }
      auto: boolean
      overflow?: boolean
    }) {
      const msg = yield* session.updateMessage({
        id: MessageID.ascending(),
        role: "user",
        model: input.model,
        sessionID: input.sessionID,
        agent: input.agent,
        time: { created: Date.now() },
      })
      yield* session.updatePart({
        id: PartID.ascending(),
        messageID: msg.id,
        sessionID: msg.sessionID,
        type: "compaction",
        auto: input.auto,
        overflow: input.overflow,
      })
    })

    return Service.of({
      isOverflow,
      prune,
      process: processCompaction,
      create,
    })
  }),
)

export const node = LayerNode.make({
  service: Service,
  layer: layer,
  deps: [
    Config.node,
    Session.node,
    Agent.node,
    Plugin.node,
    SessionProcessor.node,
    Provider.node,
    EventV2Bridge.node,
    RuntimeFlags.node,
    Database.node,
  ],
})

export * as SessionCompaction from "./compaction"
