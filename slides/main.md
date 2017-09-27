name: inverse
class: center, middle, inverse
layout: true
title: 2017fnlp-speech


---
class: titlepage, no-number

# End-to-End Speech Recognition<br />with Deep Recurrent Neural Networks
## .author[Sangho Lee]
### .gray.small[Sept. 28, 2017]

---
layout: false

## Papers Covered
- A. Graves et al., Speech Recognition with Deep Recurrent Neural Networks, in ICASSP 2013.
  - First time to apply deep RNN architectures to speech recognition
  - An enhancement to RNN transducer `[1]` where a FC layer is added
  - Phoneme Recognition on the TIMIT `[2]`


- A. Graves and N. Jaitly, Towards End-to End Speech Recognition with Recurrent Neural Networks
  - Expected transcription loss, a moditification to Connectionist Temporal Classification (CTC) `[3]`
  - Character / Word level transcription on the WSJ corpus

---
# Traditional Role of DNN for Speech Recogntion - Feature Extractor

- As a Feature Extractor
- Using GMM-HMM models for speech recognition

.center.img-60[![](images/DNN-GMM-HMM.png)]
.footnote[Image Credit: https://telecombcn-dl.github.io/2017-dlsl/]
---
# Traditional Role of DNN for Speech Recogntion - Feature Extractor

- As a Feature Extractor
- Using GMM-HMM models for speech recognition

.center.img-60[![](images/acoustic-GMM.jpg)]
.footnote[Image Credit: https://telecombcn-dl.github.io/2017-dlsl/]
---
# Traditional Role of DNN for Speech Recognition - DNN-HMM

- DNN-HMM hybrid
- Output : posterior probabilities of HMM states

.center.img-60[![](images/DNN-HMM.png)]
.footnote[Image Credit: https://telecombcn-dl.github.io/2017-dlsl/]
---
# Traditional Role of DNN for Speech Recognition - DNN-HMM

- DNN-HMM Hybrid
- Output : posterior probabilities of HMM states

.center.img-60[![](images/Hybrid-DNN.jpg)]
.footnote[Image Credit: https://telecombcn-dl.github.io/2017-dlsl/]
---
# Disadvantages of Traditional Models
- Networks as a single component in a complex pipeline
  - Need input feature extractrion (MFCC, PLP)
  - Need expertise for preprocessing (pronunciation dictionary, state-tying)
- Objective function far from the true transcription measure
- Forced alignment to make training targets
  - determined by HMM
  - lead to Sayre's paradox
---
# Towards End-to-End Training : Deep Bidirectional LSTMs
- Deep (Stacked) RNNs
  - crucial factor to model performance
- Bidirectional LSTMs
  - utterances are transcribed at once
  - bidrectional long-range context
- LSTM augmented by "peephole conections"
  - direct connection from a cell state to gate layers
  - good for precise timing tasks `[4]`
---
# Towards End-to-End Training : Deep Bidirectional LSTMs
.center.img-100[![](images/peephole.png)]

---
# Towards End-to-End Training : CTC
- Connectionist Temporal Classification `[3]`
  - an objective function that allows an RNN to be trained without any prior alignment
  - $Pr(k|t)$ : probability of emitting label k at input time step t
  - K labels + blank (null emission)

.center.img-80[![](images/emission.png)]
---
# Towards End-to-End Training : CTC
- CTC alignment $\mathbf a$
  - conditional independence assumption between each emission given the input

.center.img-50[![](images/ctcalign.png)]
---
# Towards End-to-End Training : CTC
- CTC objective : probability of the output transcription $\mathbf y$
  - $\mathcal B$ : removes repeated labels / blanks<br />
    ex) (a, -, b, -, b, c, c) -> (a, b, b, c)
  - integrating out over all possible alignments
  - computed by forward-backward algorithm

.center.img-50[![](images/ctcobjective.png)]
---
# Variants of CTC : Expected Transcription Loss
- CTC objective : maximizes only the correct target transcription
- Evaluation Measure : WER / CER / PER
  - Based on the edit distance
  - ER = $\frac{S+D+I}{N} = \frac{S+D+I}{S+D+C}$, where<br />
    S, D, I : Substituion / Deletion / Insertion,<br />
    N, C : total words / correct symbols in the reference
- Want transcriptions similar to GT (lower ER) to be more probable
---
# Variants of CTC : Expected Transcription Loss
- Loss : Expected value of an evaluation measure
  - foward/backward proprogation can be computed by Monte-Carlo sampling

.center.img-50[![](images/monteloss.png)]
---
# Variants of CTC : Expected Transcription Loss
- Loss : Expected value of an evaluation measure
  - forward/backward proprogation can be computd by Monte-Carlo sampling
.center.img-50[![](images/montegradient.png)]
---
# Variants of CTC : Expected Transcription Loss
- Loss : Expected value of an evaluation measure
  - Reusable alignment samples by conditional independence : reduce noise due to the loss variance
  - Derivative of $y_{t}^{k}$ : lead to only t-th label changes<br />
  ex) "WTRD ERROR RATE" -> "WORD ERROR RATE"
  - Expected transcription loss only used to retrain a trained network with CTC
---
# Variants of CTC : RNN Transducer
- Joint model of a language model and CTC (acoustic model) `[1]`
  - $Pr(k|t, u)$ : probability of emitting label k at input step t and output step u
  - Originally computed by a separate acoustic distribution $Pr(k|t)$ (CTC) and a linguistic distribution $Pr(k|u)$ (LM)
  - $Pr(k|t, u) \propto Pr(k|t)Pr(k|u)$
---
# Variants of CTC : RNN Transducer
- Joint model of a language model and CTC (acoustic model) `[1]`
.center.img-45[![](images/transducer.png)]
---
# Variants of CTC : RNN Transducer
- Improve capacity by an additional FC layer
  - Experimentally found that the number of deletion errors reduced
  - Pre-train each component and fine-tune the whole network
.center.img-70[![](images/additional_fc.png)]
---
# Decoding
- Best path decoding
  - Based on the assumption that the most probable path will correspond to the most probable labelling

.center.img-70[![](images/bestpath.png)]
---
# Decoding
- Prefix search decoding
.center.img-50[![](images/prefix.png)]
---
# Decoding
- Beam Search
.center.img-50[![](images/beam.png)]
- $Pr(\mathbf y, t)$ : probability of the output $\mathbf y$ from partial alignment of length t
- $Pr(\mathbf y, t)= Pr^{+}(\mathbf y, t) + Pr^{-}(\mathbf y, t)$
- $Pr^{+}(\mathbf y, t)$ : probability of $\mathbf y$ whose t-th label is not blank
- $Pr^{-}(\mathbf y, t)$ : probability of $\mathbf y$ whose t-th label is blank
---
# Decoding
- Beam Search
.center.img-50[![](images/beam.png)]
- $Pr(k, \mathbf y, t)$ : extension probability
<br />
.cen.img-60[![](images/extensionprob.png)]
- $Pr(k|\mathbf y)$ : transition probability from $\mathbf y$ to $\mathbf y + k$
---
# Experiments
- With Expected Transcription Loss
  - Dataset : WSJ
  - Measure : CER / WER
  - Input : spectrograms of raw audio files, 128 dimension per frame
  - Baseline : DNN-HMM trained by Kaldi speech recognition toolkit
  - Decoding : Beam search
.footnote[https://github.com/kaldi-asr/kaldi]
---
# Experiments
- With Expected Transcription Loss
.center.img-60[![](images/wsj.png)]
- power model learned by RNN
- implicitly learned language model
---
# Experiments
- With Expected Transcription Loss

.center.img-90[![](images/waveform.png)]
---
# Experiments
- With RNN Transducer
  - Dataset : TIMIT
  - Measure : PER
  - Input : MFCC feature, 123 dimension per frame
  - Regularisation : early stopping / Gaussian weight noise during traning
  - Decoding : Beam search with width 100
---
# Experiments
- With RNN Transducer

.center.img-80[![](images/TIMIT.png)]
---
# References
.small[`[1]` A. Graves, Sequence Transduction with Recurrent Neural Networks, in ICML 2012 Representation Learning Workshop<br />]
.small[`[2]` DARPA-ISTO, The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus (TIMIT), 1990<br />]
.small[`[3]` A. Graves et al., Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks, in ICML 2006.<br />]
.small[`[4]` F. Gers et al., Learning Precise Timing with LSTM Recurrent Networks, in Journal of Machine Learning Research 2002.<br />]
<!-- vim: set ft=markdown: -->
