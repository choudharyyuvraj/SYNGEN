// Research figures shipped in the repo's /assets folder.
const files = import.meta.glob(
  '../../assets/{one_few_shot_v2,overview_v2,One-DM_result,English_1_2,Chinese,Chinese_3_v4,casia_v4,Japanese,indus-English,indus-English_v2,indus-Chinese,industrial-methods-v4}.png',
  { eager: true, import: 'default', query: '?url' },
)
const url = (name) => files[`../../assets/${name}`]

export const FIGURES = [
  { file: 'one_few_shot_v2.png', cat: 'Method', title: 'Few-shot vs one-shot', text: 'Earlier methods need around fifteen style references; this approach needs only one.' },
  { file: 'overview_v2.png', cat: 'Method', title: 'Model overview', text: 'Content encoder, spatial and high-frequency style encoders, style-content fusion, and the diffusion generation process.' },
  { file: 'One-DM_result.png', cat: 'English', title: 'English comparison', text: 'Same sentence in four writer styles, compared with GANwriting, HiGAN+, WordStylist, HWT and VATr.' },
  { file: 'English_1_2.png', cat: 'English', title: 'English samples', text: 'More qualitative English handwriting results.' },
  { file: 'Chinese.png', cat: 'Multilingual', title: 'Chinese', text: 'Chinese handwriting generated from one reference.' },
  { file: 'Chinese_3_v4.png', cat: 'Multilingual', title: 'Chinese, more writers', text: 'Further Chinese samples across different writers.' },
  { file: 'casia_v4.png', cat: 'Multilingual', title: 'CASIA-HWDB', text: 'Results on the CASIA Chinese handwriting dataset.' },
  { file: 'Japanese.png', cat: 'Multilingual', title: 'Japanese', text: 'Japanese handwriting generation.' },
  { file: 'indus-English.png', cat: 'Comparisons', title: 'vs. general image generators (English)', text: 'DALL·E 3, Stable Diffusion, Artbreeder and IP-Adapter given the same style sample.' },
  { file: 'indus-English_v2.png', cat: 'Comparisons', title: 'vs. general image generators (English, v2)', text: 'More comparisons with general-purpose image generators.' },
  { file: 'indus-Chinese.png', cat: 'Comparisons', title: 'vs. general image generators (Chinese)', text: 'The same comparison on Chinese characters.' },
  { file: 'industrial-methods-v4.png', cat: 'Comparisons', title: 'Industrial methods', text: 'Side-by-side with general-purpose generation tools.' },
].map((f) => ({ ...f, src: url(f.file) })).filter((f) => f.src)

export const figure = (name) => url(name)

export const CREDIT = 'Research figures from One-DM (Dai et al., ECCV 2024), the one-shot diffusion model SYNGEN builds on.'
