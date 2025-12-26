# Heterogeneous-Effects-of-Multimarket-Contact-in-Airline-Pricing

## Overview

This project studies how **multimarket contact (MMC)** affects airline ticket prices in the United States, with a particular focus on **heterogeneity across market concentration**. Rather than asking whether MMC raises prices on average, the analysis examines **when** MMC matters by allowing its pricing effect to vary with market structure.

The project constructs a carrier–market–quarter panel using U.S. Department of Transportation data and introduces an **intensity‑weighted MMC measure** based on seat‑share overlap across markets. Empirical results show little average effect of MMC on fares, but a **positive and statistically significant interaction between MMC and market concentration**, indicating that MMC is associated with higher prices only in sufficiently concentrated markets.

---

## Research Questions

* Does multimarket contact affect airline fares on average?
* Does the pricing effect of multimarket contact depend on market concentration?
* Can an intensity‑weighted measure of MMC reconcile mixed findings in the existing literature?

---

## Data Sources

* **DB1B Airline Origin and Destination Survey**
  Used to construct market‑quarter average fares (undirected city‑pair markets).

* **T‑100 Domestic Segment Data**
  Used to measure carrier capacity (seats, departures) and construct carrier presence, market shares, concentration (HHI), and multimarket contact.

All markets are defined as **undirected origin–destination city pairs** to ensure consistency across datasets.

---

## Key Methodology

### Multimarket Contact Measure

* Constructs a **seat‑share‑weighted MMC index** at the carrier–market–quarter level.
* Captures overlap with the same rivals across other markets, weighting each overlap by the rival’s importance in the focal market.
* MMC is **lagged by one quarter** in regressions to mitigate simultaneity concerns.

### Empirical Strategy

* Outcome: log average fare at the market–quarter level.
* Estimation: fixed‑effects panel regressions with

  * **Carrier–market fixed effects**
  * **Quarter fixed effects**
* Inference: **market‑level clustered standard errors**.
* Main specification allows MMC effects to vary with market concentration (HHI).

---

## Main Findings

* **No strong average effect** of multimarket contact on fares.
* **Significant heterogeneity**: the effect of MMC increases with market concentration.
* In competitive markets, MMC has little or negative association with fares.
* In highly concentrated markets, greater MMC is associated with **higher fares**, consistent with conditional multimarket forbearance.

---

## Outputs

* **Main tables**: baseline and interaction regressions
* **Appendix tables**: robustness checks and alternative specifications
* **Figures**: marginal effect of MMC by concentration and descriptive concentration plots

---

## Notes and Limitations

* Prices are measured at the **market‑quarter level**, not carrier‑specific fares.
* Results should not be interpreted as causal evidence of collusion.
* MMC captures exposure to repeated rivalry, not explicit coordination.

---

## Author

This project was developed as part of an independent empirical research study on airline competition and multimarket contact.

---

## License

This project is for academic and educational purposes only.
