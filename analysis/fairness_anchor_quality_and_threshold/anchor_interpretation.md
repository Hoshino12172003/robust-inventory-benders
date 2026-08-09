# Anchor and threshold interpretation

All 20 baseline anchors are certified within the frozen 1e-4 relative tolerance. The bound g_i limits how far each anchor can lie above the unknown exact robust optimum; the corresponding conservative rho=0.01 effective increment is `(1.01)/(1-g_i)-1`. These bounds are orders of magnitude below the observed service-level changes, so anchor conservatism cannot explain the rho=0 to 0.01 jump.

The seed-level table reports budget use, continuous inventory movement, and discrete warehouse-opening switches. Conclusions are based on all ten seeds per scale, not selected cases.
