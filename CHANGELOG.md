# CHANGELOG



## v0.2.0 (2024-04-21)

### Chore

* chore: pipeline adjustments ([`2ca8d45`](https://github.com/patrk/pyPTE/commit/2ca8d4516d7730620fda0c526d32852092d5f5a5))

### Feature

* feat: add new tests, type hinting, resolve issues (#17) ([`945ce2e`](https://github.com/patrk/pyPTE/commit/945ce2e7d3d11bde238965211d5d7f41a4f433cf))

### Fix

* fix: (crucial) swapped dimensions in PTE calculation

* fix: (crucial) swapped dimensions in PTE calculation

* fix: predictions calculation using n_samples instead of m_channels ([`3ca8e9b`](https://github.com/patrk/pyPTE/commit/3ca8e9b0fa8e173a7b6bf31b0a78e45ef0e4a2ed))

* fix: (crucial) swapped dimensions in PTE calculation (#28) ([`eeb59fc`](https://github.com/patrk/pyPTE/commit/eeb59fc96d586aac389028f32687487bd754b964))

* fix: swapped axes of time series and phase array (#27) ([`e018213`](https://github.com/patrk/pyPTE/commit/e018213c473bc2ef873f7b4759b50e467c70c464))

* fix: get_phase swapped dimensions, get_phase wrongly expected the time_series of shape (n_samples, m_channels) instead of (m_channels, n_samples) (#26) ([`4bf6b5e`](https://github.com/patrk/pyPTE/commit/4bf6b5e83c48c12b302750d2095297b4772e8a56))

* fix: adjust zero-crossing detection: shift by 1, include all data points (#19) ([`4f9d2b4`](https://github.com/patrk/pyPTE/commit/4f9d2b4dbc26c843ea7a51c1c1e92921e60b74c4))

* fix: actions pipeline (#18) ([`aac8222`](https://github.com/patrk/pyPTE/commit/aac8222868b7b46573e338d366cdda66ecb63ec2))

### Unknown

* Bump idna from 3.6 to 3.7 (#24) ([`37932db`](https://github.com/patrk/pyPTE/commit/37932db209488852d5b35a42c096bed5ec3e6519))

* Patrk patch 1 (#23)

* Update README.md ([`26dfddc`](https://github.com/patrk/pyPTE/commit/26dfddc85a6c4dea5f5076f48dc02ac01f6aa356))


## v0.1.0 (2024-04-03)

### Unknown

* Merge pull request #16 from patrk/dependabot/pip/scipy-1.11.1 ([`fcac46c`](https://github.com/patrk/pyPTE/commit/fcac46c7ad0798401c4379b048a5e3c400c2a5ca))

* Bump scipy from 1.7.1 to 1.11.1

Bumps [scipy](https://github.com/scipy/scipy) from 1.7.1 to 1.11.1.
- [Release notes](https://github.com/scipy/scipy/releases)
- [Commits](https://github.com/scipy/scipy/compare/v1.7.1...v1.11.1)

---
updated-dependencies:
- dependency-name: scipy
  dependency-type: direct:production
...

Signed-off-by: dependabot[bot] &lt;support@github.com&gt; ([`7e7f5e8`](https://github.com/patrk/pyPTE/commit/7e7f5e816cf750bcd674b751deea9d0e673f2d5e))

* Merge pull request #15 from patrk/14-the-definition-of-bin-width

Use number of samples in bin size calculation ([`15d3ada`](https://github.com/patrk/pyPTE/commit/15d3ada585e0bc3c246b9bb4aee7a698d0504cbc))

* Use number of samples in bin size calculation ([`45c52a5`](https://github.com/patrk/pyPTE/commit/45c52a5b2a75584fb714f96afadd563517728e47))

* Merge pull request #13 from patrk/patrk-patch-1

Update README.md ([`7e3ddc0`](https://github.com/patrk/pyPTE/commit/7e3ddc0f6c520b51fd3873b5ac34d9caa7110bf6))

* Update README.md ([`6091605`](https://github.com/patrk/pyPTE/commit/60916054c5c8ebddfa7e6f8fcdf8a062fc4c4224))

* refined README.md equations ([`0d59bc0`](https://github.com/patrk/pyPTE/commit/0d59bc08824ff4a07e09440be25761cf94d6e1f3))

* refined README.md equations ([`dcdfbf2`](https://github.com/patrk/pyPTE/commit/dcdfbf2d9b9e30e4725b59a516f04689aa906726))

* refined README.md
specified versions in requirements.txt ([`bd644f3`](https://github.com/patrk/pyPTE/commit/bd644f32b07d8304bce171f97e456f1c171231c9))

* refined README.md ([`feb11db`](https://github.com/patrk/pyPTE/commit/feb11dbdf37ca47ad96c003a09bbe32492035491))

* Merge pull request #10 from Ku4eruk/master

compute_PTE matrix dimension changed ([`152a5bb`](https://github.com/patrk/pyPTE/commit/152a5bbcddf2cc47f2b01e78bcd5d2b485681d94))

* compute_PTE matrix dimension changed ([`5b1c3a8`](https://github.com/patrk/pyPTE/commit/5b1c3a8cdf0d1321cd10417934a355b6d6b09dfb))

* api-doc ([`2e165d9`](https://github.com/patrk/pyPTE/commit/2e165d9d9bebfa1c01477190e762177ba3956f10))

* api-doc ([`70f240a`](https://github.com/patrk/pyPTE/commit/70f240ac644d655a4f25db2702c24e92bec97d39))

* apidoc ([`edf9ff8`](https://github.com/patrk/pyPTE/commit/edf9ff84ad1a239073206540d35f2507f52d7e16))

* mne_tools doc ([`4fee056`](https://github.com/patrk/pyPTE/commit/4fee05657efa5a610b22a6582dc476a8cb9e727d))

* mne_tools doc ([`d49cbb4`](https://github.com/patrk/pyPTE/commit/d49cbb4e1bfaa6c47ab036e68fa31ff63fdfeeab))

* mne_tools doc ([`cdc5c60`](https://github.com/patrk/pyPTE/commit/cdc5c608f24e8e5cd4b26e2a8a9a4dc3e008230d))

* Merge remote-tracking branch &#39;origin/master&#39; ([`1d45cbf`](https://github.com/patrk/pyPTE/commit/1d45cbfa271613bd834097331cff2056e2631482))

* mne_tools doc ([`30f9a7b`](https://github.com/patrk/pyPTE/commit/30f9a7b3024193d5acf80eb122a598f7580636a4))

* Update README.rst ([`372e9d3`](https://github.com/patrk/pyPTE/commit/372e9d3d66be2db6886ccffeff8b83a47127fd14))

* mne_tools doc ([`2ec652c`](https://github.com/patrk/pyPTE/commit/2ec652c97758e21f760d4c4d9475e85193228dc2))

* mne_tools doc ([`39fd656`](https://github.com/patrk/pyPTE/commit/39fd6561be88bccc5cfae58b1f607d76f1200e94))

* clean up neural mass model ([`db1f5b5`](https://github.com/patrk/pyPTE/commit/db1f5b5a6b56276a164787d03e60690d74f7803b))

* fixed bug in multiprocessing ([`f99a6aa`](https://github.com/patrk/pyPTE/commit/f99a6aa8b4cb210d3e3b2e38fae492575f479678))

* bug in stats fixed ([`2dcf3da`](https://github.com/patrk/pyPTE/commit/2dcf3da406a440460006cb8c021b0a74bbcab419))

* Merge branch &#39;master&#39; of https://github.com/patrk/pyPTE ([`9558d5a`](https://github.com/patrk/pyPTE/commit/9558d5ad689658d19d62070afba62dcd884ec5f5))

* docs ([`cb3f783`](https://github.com/patrk/pyPTE/commit/cb3f78328e86bdc061301ec69a3225c586ed9c23))

* docs ([`c619210`](https://github.com/patrk/pyPTE/commit/c619210c4f5dcae9cd15aaec845ca841c060e646))

* Update requirements.txt ([`15b3121`](https://github.com/patrk/pyPTE/commit/15b3121869bcc33ffef22a80ae5c40f1bee56d62))

* Create requirements.txt ([`69cba9c`](https://github.com/patrk/pyPTE/commit/69cba9c52a5e7d9625dd87ab5488338f790b0658))

* docs ([`41d8e65`](https://github.com/patrk/pyPTE/commit/41d8e6530de70f350933451186d4b966ce1f4534))

* Create README.rst ([`a625a11`](https://github.com/patrk/pyPTE/commit/a625a11fd42f5b8c547fcf90e27f525c8ba15faf))

* refactor ([`ae52140`](https://github.com/patrk/pyPTE/commit/ae52140c9176527f513e37594c0c7c7aa58645ce))

* wilcoxon test on PTE ([`b2ee93e`](https://github.com/patrk/pyPTE/commit/b2ee93e2f1e8b435b5b98126bb5d0a8564560190))

* interpolate missing mne channels ([`a984e94`](https://github.com/patrk/pyPTE/commit/a984e9423a95aece91b557d82b7f2bfb4715a026))

* test ([`b438815`](https://github.com/patrk/pyPTE/commit/b438815018e69af5ad59a2d2d7d2e6b61a788518))

* cleaning up ([`dc5df57`](https://github.com/patrk/pyPTE/commit/dc5df57cd05ab2584c3d5926bbfa68cfc684a2c3))

* docs ([`11181a5`](https://github.com/patrk/pyPTE/commit/11181a51377b94517ff87ce39b668c15e865049c))

* cleaning up ([`fa3da91`](https://github.com/patrk/pyPTE/commit/fa3da919b71d213da49190b78530028199eb2502))

* docs ([`a9507be`](https://github.com/patrk/pyPTE/commit/a9507be19e780fdb73a8dc600ac84d52e7d873f5))

* docs ([`5620ab6`](https://github.com/patrk/pyPTE/commit/5620ab6211b518f58a8c4eb893b5ce88b95b6d23))

* docs ([`0537be8`](https://github.com/patrk/pyPTE/commit/0537be8759642885d080b2d0f18ad5fce8b212cb))

* project restructuring ([`7e18306`](https://github.com/patrk/pyPTE/commit/7e18306b27ab847c18f45a177c59659cf7d8e3c1))

* cleaning up ([`27bfd60`](https://github.com/patrk/pyPTE/commit/27bfd601404e7f56613f6f692459b042d0503ef0))

* cleaning up ([`41d3d2e`](https://github.com/patrk/pyPTE/commit/41d3d2e9dd9040d1bc75fe8c0e16b0f46e3a1b03))

* cleaning up ([`5c1f890`](https://github.com/patrk/pyPTE/commit/5c1f890881232c2b54382d185333b4e690af2136))

* initial commit ([`bbb069e`](https://github.com/patrk/pyPTE/commit/bbb069e0f32fe9f28da9bef9aa7471dbe3003cde))

* .gitignore ([`6a286c6`](https://github.com/patrk/pyPTE/commit/6a286c62f75e9436ec6438841c455c7917e97cba))

* Initial commit ([`139fed5`](https://github.com/patrk/pyPTE/commit/139fed58b71652c549b0e39a4d9d06bd3ec02cd5))
