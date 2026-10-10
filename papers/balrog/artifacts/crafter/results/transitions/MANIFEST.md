# results/transitions/ MANIFEST

Not committed to git — full per-step transition logs (obs, action, reward, done; gzipped JSONL) for the two Crafter conditions (A/B, 10 episodes each) plus 18 targeted exploration probes, quarantined as the source-blind model-induction leg's input corpus. Per the repo's size policy, raw transition datasets are kept off the artifact branch; this manifest makes every file verifiable (name, size, sha256) without bloating the repo.

- **File count:** 38
- **Total size:** 728,945 bytes (~0.73 MB, gzip-compressed)
- **VM path:** `/data/doh/teams/researchy/work/fable_crafter/results/transitions/`
- **Sample episode (included in this artifact set):** `SAMPLE/explore_probe_no_prereq.jsonl.gz`
  (source: `results/transitions/explore_probe_no_prereq.jsonl.gz`, 6166 bytes, sha256 `a4d91697d0d5b57d935bb4bdfa16a22e5573235e0f381904623a42cd3b3f8fc2`)

## Files (path relative to `results/transitions/`, size in bytes, sha256)

| path | size | sha256 |
|---|---|---|
| `A_seed9001.jsonl.gz` | 19384 | `ff35a1015a0f89ea9b53d9a191d0d8da60af9911ee1e2360e6154cf34beddf4f` |
| `A_seed9002.jsonl.gz` | 19407 | `e54a0ba450444a62fe30d49cfe94b4adad762ef0a39e4aaa11bc940b9dba295c` |
| `A_seed9003.jsonl.gz` | 41514 | `d89cee0021e7e955877071410336ae501e7ea65634f6ecf466830dd20c8edd7e` |
| `A_seed9004.jsonl.gz` | 11121 | `94c9c27c7c5fb43ab9dce591bc5dd2e89d11458b27b8d5138734a41c4e7e3897` |
| `A_seed9005.jsonl.gz` | 14612 | `b521a6b5103af2f4a2b8131a5cb149f55900d4b0e6ac3761cfa01647a4ac730d` |
| `A_seed9006.jsonl.gz` | 19752 | `d0a436c8973e2d9c62daf518c4511283096f9b14215daf1de29f7c836f66f454` |
| `A_seed9007.jsonl.gz` | 19647 | `470588edb01a9d87a978a8d62ac64cb1c411a24ac36e4f407943eaf0b4728b2b` |
| `A_seed9008.jsonl.gz` | 27323 | `e8198c0c5c2f7ad1bbbefdc059fabb27085fd2b05356e8bad701ec078b7b111d` |
| `A_seed9009.jsonl.gz` | 37763 | `238d41f0b93468f8d7d7de51cc87f80688811eaf74d47c7a153c1138f8623c6e` |
| `A_seed9010.jsonl.gz` | 34570 | `5eef1e8ebaa5f0b90c6dd91ada1c32b988feaba3becbad834b298dccf6fca0e2` |
| `B_seed9001.jsonl.gz` | 29181 | `a99c784be0a8ac47f88849e2f7284243c90203341dc22ff1844dfdf69354d1ef` |
| `B_seed9002.jsonl.gz` | 22627 | `f413d23fcecce607b1f30cf6d8660bf5c545462e9d261dd84eace6626e289d76` |
| `B_seed9003.jsonl.gz` | 40096 | `eb4daaed97fa629d10a628222914644a8fc18b419434c981176f18c89bd22481` |
| `B_seed9004.jsonl.gz` | 15971 | `97bca18152242bee453c622e1ec9e58c70a0fab5dc13d619a876bb59ad4390e1` |
| `B_seed9005.jsonl.gz` | 35515 | `dd9ab5b55f29b2f7e2c8774e15e6265d40985f211c4d47653c5f80484771ef05` |
| `B_seed9006.jsonl.gz` | 25735 | `d748f1fd258f03fe0f156d09cf61f07b82aed6f8d158bba61b08d3ecc6156125` |
| `B_seed9007.jsonl.gz` | 26596 | `320612a00fff3020378060b7664d59ca282f1cf6904a7205895bd52bb0b5f99c` |
| `B_seed9008.jsonl.gz` | 22243 | `b080e90f7ac373eae57bc6ac1aba6e7f0a0f07bb8a94248dc5cb2114efa174a7` |
| `B_seed9009.jsonl.gz` | 13863 | `b6916f782f676d5c8e287b2fee5be7a20456d202518cf09c53dd162a86c1fc4a` |
| `B_seed9010.jsonl.gz` | 14269 | `2ac750d3b9b28ccfd25942b2fbf4aaba417ceb3636871ff2ff55f596f5b463a7` |
| `explore_iron_smith.jsonl.gz` | 15833 | `65733257021d792f3db22a8daeeacc7eb82fda04f6fa3519e6fb9a5fd0432c23` |
| `explore_iron_tour.jsonl.gz` | 18184 | `86bce2e5b639258c516cf1f09368723fe3c3e24c3b97581933423568b6605a4d` |
| `explore_lava_death.jsonl.gz` | 12589 | `6bc7ad1d12ed85598ab5a7383f011c78dac70ec8345218f12ace37d6e6108918` |
| `explore_noop_starve.jsonl.gz` | 10507 | `17d180fd6f0ce08922a1d8cefc33d0cfd4272344f8156251f4388b254d82a928` |
| `explore_ore_tour_5011.jsonl.gz` | 14395 | `7f41b3553bba5f03e22f1cf8311bc6b99b3e90d86bc85e40af17752b24d8d892` |
| `explore_ore_tour_5111.jsonl.gz` | 22672 | `c26c44884b49ff4d4902f09557a22d7e6d1a68358c1c262f1c1fbe40233d6aa9` |
| `explore_plant_lifecycle_5110.jsonl.gz` | 16041 | `602cfcabe7c1b8053f04a4d9c35237c5931cf34f19a835d53e0ef1d5e276f632` |
| `explore_plant_lifecycle_5310.jsonl.gz` | 14902 | `0162c5965bb08fcb25935a2806012b9af36f25826d76341ea212f16ecb4defef` |
| `explore_plant_lifecycle_5410.jsonl.gz` | 8121 | `37c0a02411dff867cd309236ee3a84d830c5a9b321b2147914207f86b9dec60d` |
| `explore_plant_lifecycle_5510.jsonl.gz` | 13706 | `03c9559e8f23404e7ade2aa16bbf4077dd7bc9db883d7e8314182f2d08ea5217` |
| `explore_plant_ripe_forced.jsonl.gz` | 14791 | `15925b427656f59debc4f1624055d7baae2f5349c1cff8f66689bf449dad04e1` |
| `explore_probe_boosted.jsonl.gz` | 12074 | `a70f600e2cd4c9aa0db78fa1a7ec1e0b2f8373a6bb32a3ac5e7f05d7e03ab1da` |
| `explore_probe_no_prereq.jsonl.gz` | 6166 | `a4d91697d0d5b57d935bb4bdfa16a22e5573235e0f381904623a42cd3b3f8fc2` |
| `explore_random_1.jsonl.gz` | 11558 | `f9f2525bad6508447e3e642f185ee508cdf9bc9aaa80d1353cfbba6993228cd6` |
| `explore_random_2.jsonl.gz` | 11899 | `398e859731cbd4422d0c4078cae1b749e9f0e6dd4ac525da8795f067b1c1d399` |
| `explore_random_3.jsonl.gz` | 10311 | `347a31a6f80db6710dd432bf72dc9a79c11fbdad78321535892b802331ce0c34` |
| `explore_skeleton_death.jsonl.gz` | 11604 | `1b95e01d5cb606ddf78e43b3572db622bfe0050c35880576c83ca1fee5e28997` |
| `explore_sleep_attack.jsonl.gz` | 12403 | `d3321110e1093f2ae75daa00ed51e6bd0ed7091b995e426dbe2a8975830a083b` |
