/**
 * GPU presets for the REAP cost calculator.
 *
 * The TFLOPS figures are approximate dense tensor throughput, derived from the
 * vendor's published AI TOPS where a dense figure is not listed, using
 * BF16 dense = AI TOPS / 8 and FP8 dense = AI TOPS / 4. Vendors quote AI TOPS
 * with FP4 sparsity, so the dense figures here are deliberately conservative.
 *
 * Real achieved throughput depends on the kernel, the framework, and the
 * precision the model is stored in. The model factory utilisation (MFU) input
 * in the calculator absorbs that gap, which is why these numbers only need to
 * be right to within a factor of about two.
 *
 * fp8DenseTflops is null where the architecture has no FP8 tensor support.
 * That is Ampere (RTX 30xx) and RDNA 2 and RDNA 3 (RX 60xx and RX 70xx), so a
 * REAP run on those cards must be calibrated in BF16 or FP16.
 */

export type GpuVendor = 'nvidia' | 'amd'

export interface GpuSpec {
  id: string
  label: string
  vendor: GpuVendor
  generation: string
  vramGiB: number
  /** Memory bandwidth in GB/s, decimal gigabytes. */
  bandwidthGBs: number
  /** Dense BF16 tensor throughput in TFLOPS. */
  bf16DenseTflops: number
  /** Dense FP8 tensor throughput in TFLOPS, or null when unsupported. */
  fp8DenseTflops: number | null
  note: string
}

export const GPU_PRESETS: GpuSpec[] = [
  // Datacenter, current generation.
  {
    id: 'b300',
    label: 'B300 (Blackwell Ultra)',
    vendor: 'nvidia',
    generation: 'Blackwell Ultra',
    vramGiB: 288,
    bandwidthGBs: 8000,
    bf16DenseTflops: 3500,
    fp8DenseTflops: 7000,
    note: '288 GB HBM3e. The largest single-GPU budget here.',
  },
  {
    id: 'b200',
    label: 'B200',
    vendor: 'nvidia',
    generation: 'Blackwell',
    vramGiB: 192,
    bandwidthGBs: 8000,
    bf16DenseTflops: 2250,
    fp8DenseTflops: 4500,
    note: '192 GB HBM3e.',
  },
  {
    id: 'h200',
    label: 'H200',
    vendor: 'nvidia',
    generation: 'Hopper',
    vramGiB: 141,
    bandwidthGBs: 4800,
    bf16DenseTflops: 989,
    fp8DenseTflops: 1979,
    note: '141 GB HBM3e. The reference point for most published REAP runs.',
  },
  {
    id: 'h100',
    label: 'H100 SXM',
    vendor: 'nvidia',
    generation: 'Hopper',
    vramGiB: 80,
    bandwidthGBs: 3350,
    bf16DenseTflops: 989,
    fp8DenseTflops: 1979,
    note: '80 GB HBM3. The PCIe card is slower on memory: about 2.0 TB/s.',
  },
  {
    id: 'rtx-pro-6000',
    label: 'RTX PRO 6000 Blackwell',
    vendor: 'nvidia',
    generation: 'Blackwell',
    vramGiB: 96,
    bandwidthGBs: 1792,
    bf16DenseTflops: 500,
    fp8DenseTflops: 1000,
    note: '96 GB GDDR7. The largest single workstation card here.',
  },

  // RTX 50 series, Blackwell. AI TOPS quoted with FP4 sparsity.
  {
    id: 'rtx-5090',
    label: 'RTX 5090',
    vendor: 'nvidia',
    generation: 'Blackwell',
    vramGiB: 32,
    bandwidthGBs: 1792,
    bf16DenseTflops: 419,
    fp8DenseTflops: 838,
    note: '32 GB GDDR7. 3352 AI TOPS.',
  },
  {
    id: 'rtx-5080',
    label: 'RTX 5080',
    vendor: 'nvidia',
    generation: 'Blackwell',
    vramGiB: 16,
    bandwidthGBs: 960,
    bf16DenseTflops: 225,
    fp8DenseTflops: 450,
    note: '16 GB GDDR7. 1801 AI TOPS.',
  },
  {
    id: 'rtx-5070-ti',
    label: 'RTX 5070 Ti',
    vendor: 'nvidia',
    generation: 'Blackwell',
    vramGiB: 16,
    bandwidthGBs: 896,
    bf16DenseTflops: 176,
    fp8DenseTflops: 351,
    note: '16 GB GDDR7. 1406 AI TOPS.',
  },
  {
    id: 'rtx-5070',
    label: 'RTX 5070',
    vendor: 'nvidia',
    generation: 'Blackwell',
    vramGiB: 12,
    bandwidthGBs: 672,
    bf16DenseTflops: 123,
    fp8DenseTflops: 247,
    note: '12 GB GDDR7. 988 AI TOPS.',
  },
  {
    id: 'rtx-5060-ti',
    label: 'RTX 5060 Ti 16 GB',
    vendor: 'nvidia',
    generation: 'Blackwell',
    vramGiB: 16,
    bandwidthGBs: 448,
    bf16DenseTflops: 95,
    fp8DenseTflops: 190,
    note: '16 GB GDDR7. An 8 GB variant exists and will not fit a large block.',
  },
  {
    id: 'rtx-5060',
    label: 'RTX 5060',
    vendor: 'nvidia',
    generation: 'Blackwell',
    vramGiB: 8,
    bandwidthGBs: 448,
    bf16DenseTflops: 77,
    fp8DenseTflops: 153,
    note: '8 GB GDDR7. 614 AI TOPS.',
  },

  // RTX 40 series, Ada Lovelace.
  {
    id: 'rtx-4090',
    label: 'RTX 4090',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 24,
    bandwidthGBs: 1008,
    bf16DenseTflops: 165,
    fp8DenseTflops: 330,
    note: '24 GB GDDR6X. 1321 AI TOPS.',
  },
  {
    id: 'rtx-4080-super',
    label: 'RTX 4080 Super',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 16,
    bandwidthGBs: 736,
    bf16DenseTflops: 104,
    fp8DenseTflops: 209,
    note: '16 GB GDDR6X.',
  },
  {
    id: 'rtx-4080',
    label: 'RTX 4080',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 16,
    bandwidthGBs: 717,
    bf16DenseTflops: 97,
    fp8DenseTflops: 195,
    note: '16 GB GDDR6X.',
  },
  {
    id: 'rtx-4070-ti-super',
    label: 'RTX 4070 Ti Super',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 16,
    bandwidthGBs: 672,
    bf16DenseTflops: 88,
    fp8DenseTflops: 176,
    note: '16 GB GDDR6X.',
  },
  {
    id: 'rtx-4070-ti',
    label: 'RTX 4070 Ti',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 12,
    bandwidthGBs: 504,
    bf16DenseTflops: 80,
    fp8DenseTflops: 160,
    note: '12 GB GDDR6X.',
  },
  {
    id: 'rtx-4070-super',
    label: 'RTX 4070 Super',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 12,
    bandwidthGBs: 504,
    bf16DenseTflops: 71,
    fp8DenseTflops: 142,
    note: '12 GB GDDR6X.',
  },
  {
    id: 'rtx-4070',
    label: 'RTX 4070',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 12,
    bandwidthGBs: 504,
    bf16DenseTflops: 58,
    fp8DenseTflops: 116,
    note: '12 GB GDDR6X.',
  },
  {
    id: 'rtx-4060-ti',
    label: 'RTX 4060 Ti 16 GB',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 16,
    bandwidthGBs: 288,
    bf16DenseTflops: 44,
    fp8DenseTflops: 88,
    note: '16 GB GDDR6. An 8 GB variant exists.',
  },
  {
    id: 'rtx-4060',
    label: 'RTX 4060',
    vendor: 'nvidia',
    generation: 'Ada Lovelace',
    vramGiB: 8,
    bandwidthGBs: 272,
    bf16DenseTflops: 30,
    fp8DenseTflops: 61,
    note: '8 GB GDDR6.',
  },

  // RTX 30 series, Ampere. No FP8 tensor support.
  {
    id: 'rtx-3090-ti',
    label: 'RTX 3090 Ti',
    vendor: 'nvidia',
    generation: 'Ampere',
    vramGiB: 24,
    bandwidthGBs: 1008,
    bf16DenseTflops: 80,
    fp8DenseTflops: null,
    note: '24 GB GDDR6X. Ampere has no FP8 path, so calibrate in BF16.',
  },
  {
    id: 'rtx-3090',
    label: 'RTX 3090',
    vendor: 'nvidia',
    generation: 'Ampere',
    vramGiB: 24,
    bandwidthGBs: 936,
    bf16DenseTflops: 71,
    fp8DenseTflops: null,
    note: '24 GB GDDR6X. Ampere has no FP8 path.',
  },
  {
    id: 'rtx-3080-ti',
    label: 'RTX 3080 Ti',
    vendor: 'nvidia',
    generation: 'Ampere',
    vramGiB: 12,
    bandwidthGBs: 912,
    bf16DenseTflops: 68,
    fp8DenseTflops: null,
    note: '12 GB GDDR6X. Ampere has no FP8 path.',
  },
  {
    id: 'rtx-3080',
    label: 'RTX 3080',
    vendor: 'nvidia',
    generation: 'Ampere',
    vramGiB: 10,
    bandwidthGBs: 760,
    bf16DenseTflops: 60,
    fp8DenseTflops: null,
    note: '10 GB GDDR6X. Ampere has no FP8 path.',
  },
  {
    id: 'rtx-3070-ti',
    label: 'RTX 3070 Ti',
    vendor: 'nvidia',
    generation: 'Ampere',
    vramGiB: 8,
    bandwidthGBs: 608,
    bf16DenseTflops: 44,
    fp8DenseTflops: null,
    note: '8 GB GDDR6X. Ampere has no FP8 path.',
  },
  {
    id: 'rtx-3070',
    label: 'RTX 3070',
    vendor: 'nvidia',
    generation: 'Ampere',
    vramGiB: 8,
    bandwidthGBs: 448,
    bf16DenseTflops: 41,
    fp8DenseTflops: null,
    note: '8 GB GDDR6. Ampere has no FP8 path.',
  },
  {
    id: 'rtx-3060-ti',
    label: 'RTX 3060 Ti',
    vendor: 'nvidia',
    generation: 'Ampere',
    vramGiB: 8,
    bandwidthGBs: 448,
    bf16DenseTflops: 32,
    fp8DenseTflops: null,
    note: '8 GB GDDR6. Ampere has no FP8 path.',
  },
  {
    id: 'rtx-3060',
    label: 'RTX 3060',
    vendor: 'nvidia',
    generation: 'Ampere',
    vramGiB: 12,
    bandwidthGBs: 360,
    bf16DenseTflops: 25,
    fp8DenseTflops: null,
    note: '12 GB GDDR6. Ampere has no FP8 path.',
  },

  // AMD datacenter.
  {
    id: 'mi300x',
    label: 'Instinct MI300X',
    vendor: 'amd',
    generation: 'CDNA 3',
    vramGiB: 192,
    bandwidthGBs: 5320,
    bf16DenseTflops: 1307,
    fp8DenseTflops: 2615,
    note: '192 GB HBM3. The largest AMD memory budget here.',
  },

  // AMD Radeon RX 9000 series, RDNA 4. FP8 tensor support is new here.
  {
    id: 'rx-9070-xt',
    label: 'Radeon RX 9070 XT',
    vendor: 'amd',
    generation: 'RDNA 4',
    vramGiB: 16,
    bandwidthGBs: 645,
    bf16DenseTflops: 195,
    fp8DenseTflops: 389,
    note: '16 GB GDDR6. 1557 INT4 AI TOPS.',
  },
  {
    id: 'rx-9070',
    label: 'Radeon RX 9070',
    vendor: 'amd',
    generation: 'RDNA 4',
    vramGiB: 16,
    bandwidthGBs: 645,
    bf16DenseTflops: 146,
    fp8DenseTflops: 291,
    note: '16 GB GDDR6.',
  },
  {
    id: 'rx-9060-xt',
    label: 'Radeon RX 9060 XT 16 GB',
    vendor: 'amd',
    generation: 'RDNA 4',
    vramGiB: 16,
    bandwidthGBs: 322,
    bf16DenseTflops: 103,
    fp8DenseTflops: 205,
    note: '16 GB GDDR6. An 8 GB variant exists.',
  },

  // AMD Radeon RX 7000 series, RDNA 3. No FP8 tensor support.
  {
    id: 'rx-7900-xtx',
    label: 'Radeon RX 7900 XTX',
    vendor: 'amd',
    generation: 'RDNA 3',
    vramGiB: 24,
    bandwidthGBs: 960,
    bf16DenseTflops: 123,
    fp8DenseTflops: null,
    note: '24 GB GDDR6. RDNA 3 has no FP8 path.',
  },
  {
    id: 'rx-7900-xt',
    label: 'Radeon RX 7900 XT',
    vendor: 'amd',
    generation: 'RDNA 3',
    vramGiB: 20,
    bandwidthGBs: 800,
    bf16DenseTflops: 103,
    fp8DenseTflops: null,
    note: '20 GB GDDR6. RDNA 3 has no FP8 path.',
  },
  {
    id: 'rx-7800-xt',
    label: 'Radeon RX 7800 XT',
    vendor: 'amd',
    generation: 'RDNA 3',
    vramGiB: 16,
    bandwidthGBs: 624,
    bf16DenseTflops: 75,
    fp8DenseTflops: null,
    note: '16 GB GDDR6. RDNA 3 has no FP8 path.',
  },
  {
    id: 'rx-7700-xt',
    label: 'Radeon RX 7700 XT',
    vendor: 'amd',
    generation: 'RDNA 3',
    vramGiB: 12,
    bandwidthGBs: 432,
    bf16DenseTflops: 70,
    fp8DenseTflops: null,
    note: '12 GB GDDR6. RDNA 3 has no FP8 path.',
  },
  {
    id: 'rx-7600',
    label: 'Radeon RX 7600',
    vendor: 'amd',
    generation: 'RDNA 3',
    vramGiB: 8,
    bandwidthGBs: 288,
    bf16DenseTflops: 44,
    fp8DenseTflops: null,
    note: '8 GB GDDR6. RDNA 3 has no FP8 path.',
  },

  // AMD Radeon RX 6000 series, RDNA 2. No FP8 tensor support.
  {
    id: 'rx-6800-xt',
    label: 'Radeon RX 6800 XT',
    vendor: 'amd',
    generation: 'RDNA 2',
    vramGiB: 16,
    bandwidthGBs: 512,
    bf16DenseTflops: 42,
    fp8DenseTflops: null,
    note: '16 GB GDDR6. RDNA 2 has no FP8 path.',
  },
  {
    id: 'rx-6700-xt',
    label: 'Radeon RX 6700 XT',
    vendor: 'amd',
    generation: 'RDNA 2',
    vramGiB: 12,
    bandwidthGBs: 384,
    bf16DenseTflops: 26,
    fp8DenseTflops: null,
    note: '12 GB GDDR6. RDNA 2 has no FP8 path.',
  },
  {
    id: 'rx-6600',
    label: 'Radeon RX 6600',
    vendor: 'amd',
    generation: 'RDNA 2',
    vramGiB: 8,
    bandwidthGBs: 224,
    bf16DenseTflops: 18,
    fp8DenseTflops: null,
    note: '8 GB GDDR6. RDNA 2 has no FP8 path.',
  },
]

const GPU_BY_ID = new Map(GPU_PRESETS.map((gpu) => [gpu.id, gpu]))

export function findGpu(id: string): GpuSpec | undefined {
  return GPU_BY_ID.get(id)
}

/** Vendors in the order they should appear in a grouped picker. */
export const GPU_VENDORS: GpuVendor[] = ['nvidia', 'amd']

export const GPU_VENDOR_LABELS: Record<GpuVendor, string> = {
  nvidia: 'NVIDIA',
  amd: 'AMD',
}

/** The default selection, since it is the card this site was built around. */
export const DEFAULT_GPU_ID = 'rtx-5090'
