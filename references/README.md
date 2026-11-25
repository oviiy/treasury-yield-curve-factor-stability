# Reference paper

**Liangjun Su and Xia Wang (2017).** “On time-varying factor models: Estimation and testing.” *Journal of Econometrics* 198(1), 84–101. [Publisher record and DOI](https://doi.org/10.1016/j.jeconom.2016.12.004). [SMU repository record](https://ink.library.smu.edu.sg/soe_research/1779/). A reusable [BibTeX entry](SuWang2017.bib) is included here.

An earlier, author-owned [2015 SMU working-paper version](https://ink.library.smu.edu.sg/soe_research/1742/) is available from the institutional repository under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). It is an earlier version and should not be silently substituted for the 2017 journal version when checking equation or page references.

The locally available 2017 PDF is the publisher-formatted version and carries a separate rights notice. This public repository records the official paper links and citation rather than uploading that PDF without clear redistribution permission. The implementation in `code/sw_test.py` is newly written from the paper's equations and is not the authors' MATLAB code.

See also the authors' [corrigendum](https://ink.library.smu.edu.sg/soe_research/2364/) on bias terms in the estimation theory. The corrigendum states that its correction does not affect the test statistic's null distribution.

## Literature used by the serial-correlation-robust test

| Reference | Role |
| --- | --- |
| Breitung, J. and Tenhofen, J. (2011). GLS estimation of dynamic factor models. *JASA* 106(495), 1150–1166. | AR quasi-differencing (GLS) inside a factor model: the filter step. |
| Kreiss, J.-P. (1992). Bootstrap procedures for AR(∞)-processes. *Lecture Notes in Economics and Mathematical Systems* 376, 107–113. | Sieve-AR bootstrap. |
| Bühlmann, P. (1997). Sieve bootstrap for time series. *Bernoulli* 3(2), 123–148. | Sieve bootstrap with a data-chosen AR order. |
| Su, L. and Wang, X. (2020). Testing for structural changes in factor models via a nonparametric regression. *Econometric Theory* 36(6), 1127–1158. | Regression form of the local-versus-fixed comparison. |
| Shao, X. (2010). The dependent wild bootstrap. *JASA* 105(489), 218–235. | Alternative bootstrap tried and rejected (size 12–50% on the original statistic). |
| Duffee, G. R. (2011). Information in (and not in) the term structure. *Review of Financial Studies* 24(9), 2895–2934. | Hidden fourth factor; interpretation of the r = 3 vs r = 4 results. |

Full entries are in the paper's bibliography.
