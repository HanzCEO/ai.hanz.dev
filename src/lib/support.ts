/**
 * How well grounded an option is.
 *
 * A calculator never offers a choice without saying whether the hardware or the
 * architecture actually supports it, so this vocabulary is shared by every
 * picker and every tag rather than owned by whichever tool happened to need it
 * first.
 */
export type SupportLevel = 'supported' | 'untested' | 'unsupported'
