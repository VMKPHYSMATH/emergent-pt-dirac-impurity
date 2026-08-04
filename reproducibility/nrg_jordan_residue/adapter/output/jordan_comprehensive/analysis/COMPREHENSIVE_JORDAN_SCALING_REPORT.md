# Comprehensive Jordan and quartic scaling audit

NRG iterations: **4, 5, 6**.

The audit tracks full 2x2 residue-matrix pole pairs. It extrapolates the analytic variable `y=s^2`, compares multiple detuning models, fits the complex interaction response, and tests the complex quartic coefficient rather than its magnitude alone.

## Executive gates

- Numerical reliability: **PASS** (reliable tracked fraction `0.9939`).
- Jordan matrix survival: **PASS** (alignment `1`, trace fraction `0.000835671`, nilpotent mismatch `7.31791e-07`).
- Generic square-root EP perturbation: **NOT PASSED** (median complex-fit `p=1.35904`, 68% spread `1.99079`).
- Full quartic coefficient plateau: **NOT PASSED** (|log slope| `0.88017`, complex scatter `1.1182`, phase spread `0.367719`).

A gate not passing is not automatically a proof of absence; inspect branch continuity, convergence, and reliability tables before making a physics claim.

## Complex exponent ensemble

| config | model | n | q | method | Umax | p_complex | rel RMS | p_magnitude | R2 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| reference | scalar | 4 | -1 | linear_all | 0.015 | 0.367518 | 0.1161 | 0.579947 | 0.6847 |
| reference | scalar | 4 | -1 | linear_all | 0.02 | 0.247553 | 0.1258 | 0.419342 | 0.5378 |
| reference | scalar | 4 | -1 | linear_all | 0.03 | 0.2 | 0.1567 | 0.235226 | 0.2503 |
| reference | scalar | 4 | -1 | linear_all | 0.05 | 0.2 | 0.3184 | 0.420942 | 0.5079 |
| reference | scalar | 4 | -1 | linear_drop_largest | 0.015 | 0.2 | 0.2057 | 0.933228 | 0.3831 |
| reference | scalar | 4 | -1 | linear_drop_largest | 0.02 | 0.2 | 0.1937 | 0.660035 | 0.2905 |
| reference | scalar | 4 | -1 | linear_drop_largest | 0.03 | 0.2 | 0.1916 | 0.408946 | 0.1685 |
| reference | scalar | 4 | -1 | linear_drop_largest | 0.05 | 0.2 | 0.393 | 0.467837 | 0.2874 |
| reference | scalar | 4 | -1 | linear_small3 | 0.015 | 0.53772 | 0.2215 | 1.24179 | 0.63 |
| reference | scalar | 4 | -1 | linear_small3 | 0.02 | 0.380498 | 0.2197 | 0.933061 | 0.5282 |
| reference | scalar | 4 | -1 | linear_small3 | 0.03 | 0.227844 | 0.227 | 0.645318 | 0.3817 |
| reference | scalar | 4 | -1 | linear_small3 | 0.05 | 0.2 | 0.4695 | 0.53166 | 0.3794 |
| reference | scalar | 4 | -1 | quadratic_all | 0.015 | 0.2 | 0.238 | 0.907129 | 0.2809 |
| reference | scalar | 4 | -1 | quadratic_all | 0.02 | 0.2 | 0.222 | 0.627933 | 0.2046 |
| reference | scalar | 4 | -1 | quadratic_all | 0.03 | 0.2 | 0.2119 | 0.395171 | 0.1226 |
| reference | scalar | 4 | -1 | quadratic_all | 0.05 | 0.2 | 0.2447 | 0.292968 | 0.1032 |
| reference | scalar | 4 | -1 | quadratic_small4 | 0.015 | 0.806079 | 0.2052 | 1.04166 | 0.4123 |
| reference | scalar | 4 | -1 | quadratic_small4 | 0.02 | 0.568509 | 0.2054 | 0.892764 | 0.4296 |
| reference | scalar | 4 | -1 | quadratic_small4 | 0.03 | 0.357454 | 0.2145 | 0.690232 | 0.3795 |
| reference | scalar | 4 | -1 | quadratic_small4 | 0.05 | 0.2 | 0.3219 | 0.425481 | 0.2216 |
| reference | scalar | 4 | +1 | linear_all | 0.015 | 0.367524 | 0.1161 | 0.579969 | 0.6847 |
| reference | scalar | 4 | +1 | linear_all | 0.02 | 0.247556 | 0.1258 | 0.419358 | 0.5378 |
| reference | scalar | 4 | +1 | linear_all | 0.03 | 0.2 | 0.1567 | 0.235239 | 0.2503 |
| reference | scalar | 4 | +1 | linear_all | 0.05 | 0.2 | 0.3184 | 0.420951 | 0.5079 |
| reference | scalar | 4 | +1 | linear_drop_largest | 0.015 | 0.2 | 0.2057 | 0.933321 | 0.3832 |
| reference | scalar | 4 | +1 | linear_drop_largest | 0.02 | 0.2 | 0.1937 | 0.660106 | 0.2905 |
| reference | scalar | 4 | +1 | linear_drop_largest | 0.03 | 0.2 | 0.1916 | 0.409001 | 0.1685 |
| reference | scalar | 4 | +1 | linear_drop_largest | 0.05 | 0.2 | 0.393 | 0.467876 | 0.2874 |
| reference | scalar | 4 | +1 | linear_small3 | 0.015 | 0.537732 | 0.2215 | 1.24182 | 0.63 |
| reference | scalar | 4 | +1 | linear_small3 | 0.02 | 0.380507 | 0.2197 | 0.933088 | 0.5282 |
| reference | scalar | 4 | +1 | linear_small3 | 0.03 | 0.227853 | 0.227 | 0.645341 | 0.3817 |
| reference | scalar | 4 | +1 | linear_small3 | 0.05 | 0.2 | 0.4695 | 0.531676 | 0.3794 |
| reference | scalar | 4 | +1 | quadratic_all | 0.015 | 0.2 | 0.238 | 0.907171 | 0.2809 |
| reference | scalar | 4 | +1 | quadratic_all | 0.02 | 0.2 | 0.222 | 0.627966 | 0.2046 |
| reference | scalar | 4 | +1 | quadratic_all | 0.03 | 0.2 | 0.2119 | 0.395196 | 0.1226 |
| reference | scalar | 4 | +1 | quadratic_all | 0.05 | 0.2 | 0.2447 | 0.292986 | 0.1032 |
| reference | scalar | 4 | +1 | quadratic_small4 | 0.015 | 0.8061 | 0.2052 | 1.04171 | 0.4123 |
| reference | scalar | 4 | +1 | quadratic_small4 | 0.02 | 0.568524 | 0.2054 | 0.892811 | 0.4296 |
| reference | scalar | 4 | +1 | quadratic_small4 | 0.03 | 0.357468 | 0.2145 | 0.690273 | 0.3796 |
| reference | scalar | 4 | +1 | quadratic_small4 | 0.05 | 0.2 | 0.3219 | 0.425517 | 0.2216 |
| reference | scalar | 5 | -1 | linear_all | 0.015 | 1.46822 | 0.007053 | 1.4007 | 0.9993 |
| reference | scalar | 5 | -1 | linear_all | 0.02 | 1.47571 | 0.01161 | 1.4173 | 0.9993 |
| reference | scalar | 5 | -1 | linear_all | 0.03 | 1.45327 | 0.02711 | 1.42418 | 0.9995 |
| reference | scalar | 5 | -1 | linear_all | 0.05 | 1.33458 | 0.05456 | 1.39738 | 0.999 |
| reference | scalar | 5 | -1 | linear_drop_largest | 0.015 | 1.47394 | 0.008005 | 1.41807 | 0.9995 |
| reference | scalar | 5 | -1 | linear_drop_largest | 0.02 | 1.46732 | 0.01511 | 1.42823 | 0.9996 |
| reference | scalar | 5 | -1 | linear_drop_largest | 0.03 | 1.43929 | 0.03287 | 1.42852 | 0.9997 |
| reference | scalar | 5 | -1 | linear_drop_largest | 0.05 | 1.30165 | 0.06482 | 1.38962 | 0.9983 |
| reference | scalar | 5 | -1 | linear_small3 | 0.015 | 1.45648 | 0.01026 | 1.46064 | 0.9997 |
| reference | scalar | 5 | -1 | linear_small3 | 0.02 | 1.39977 | 0.02312 | 1.44375 | 0.9996 |
| reference | scalar | 5 | -1 | linear_small3 | 0.03 | 1.38146 | 0.04432 | 1.42495 | 0.9994 |
| reference | scalar | 5 | -1 | linear_small3 | 0.05 | 1.21473 | 0.0856 | 1.35837 | 0.9951 |
| reference | scalar | 5 | -1 | quadratic_all | 0.015 | 1.46894 | 0.01183 | 1.46357 | 0.9996 |
| reference | scalar | 5 | -1 | quadratic_all | 0.02 | 1.41973 | 0.02451 | 1.45047 | 0.9995 |
| reference | scalar | 5 | -1 | quadratic_all | 0.03 | 1.39493 | 0.047 | 1.4324 | 0.9994 |
| reference | scalar | 5 | -1 | quadratic_all | 0.05 | 1.23133 | 0.09203 | 1.36499 | 0.9951 |
| reference | scalar | 5 | -1 | quadratic_small4 | 0.015 | 1.43541 | 0.01489 | 1.50445 | 0.9984 |
| reference | scalar | 5 | -1 | quadratic_small4 | 0.02 | 1.33663 | 0.03349 | 1.45993 | 0.9971 |
| reference | scalar | 5 | -1 | quadratic_small4 | 0.03 | 1.32718 | 0.05871 | 1.42135 | 0.9966 |
| reference | scalar | 5 | -1 | quadratic_small4 | 0.05 | 1.14083 | 0.1158 | 1.32595 | 0.988 |
| reference | scalar | 5 | +1 | linear_all | 0.015 | 1.46821 | 0.007053 | 1.4007 | 0.9993 |
| reference | scalar | 5 | +1 | linear_all | 0.02 | 1.47571 | 0.01161 | 1.41729 | 0.9993 |
| reference | scalar | 5 | +1 | linear_all | 0.03 | 1.45327 | 0.02711 | 1.42417 | 0.9995 |
| reference | scalar | 5 | +1 | linear_all | 0.05 | 1.33458 | 0.05456 | 1.39737 | 0.999 |
| reference | scalar | 5 | +1 | linear_drop_largest | 0.015 | 1.47391 | 0.008007 | 1.41804 | 0.9995 |
| reference | scalar | 5 | +1 | linear_drop_largest | 0.02 | 1.46731 | 0.01511 | 1.4282 | 0.9996 |
| reference | scalar | 5 | +1 | linear_drop_largest | 0.03 | 1.43928 | 0.03287 | 1.4285 | 0.9997 |
| reference | scalar | 5 | +1 | linear_drop_largest | 0.05 | 1.30164 | 0.06482 | 1.3896 | 0.9983 |
| reference | scalar | 5 | +1 | linear_small3 | 0.015 | 1.45647 | 0.01026 | 1.46062 | 0.9997 |
| reference | scalar | 5 | +1 | linear_small3 | 0.02 | 1.39977 | 0.02312 | 1.44374 | 0.9996 |
| reference | scalar | 5 | +1 | linear_small3 | 0.03 | 1.38146 | 0.04432 | 1.42494 | 0.9994 |
| reference | scalar | 5 | +1 | linear_small3 | 0.05 | 1.21473 | 0.0856 | 1.35836 | 0.9951 |
| reference | scalar | 5 | +1 | quadratic_all | 0.015 | 1.46889 | 0.01184 | 1.46352 | 0.9995 |
| reference | scalar | 5 | +1 | quadratic_all | 0.02 | 1.41971 | 0.02451 | 1.45043 | 0.9995 |
| reference | scalar | 5 | +1 | quadratic_all | 0.03 | 1.39492 | 0.047 | 1.43236 | 0.9994 |
| reference | scalar | 5 | +1 | quadratic_all | 0.05 | 1.23132 | 0.09203 | 1.36496 | 0.9951 |
| reference | scalar | 5 | +1 | quadratic_small4 | 0.015 | 1.43541 | 0.01489 | 1.50443 | 0.9984 |
| reference | scalar | 5 | +1 | quadratic_small4 | 0.02 | 1.33663 | 0.03349 | 1.45991 | 0.9971 |
| reference | scalar | 5 | +1 | quadratic_small4 | 0.03 | 1.32718 | 0.05871 | 1.42134 | 0.9966 |
| reference | scalar | 5 | +1 | quadratic_small4 | 0.05 | 1.14083 | 0.1158 | 1.32594 | 0.988 |
| reference | scalar | 6 | -1 | linear_all | 0.015 | 1.93537 | 0.01386 | 1.6023 | 0.989 |
| reference | scalar | 6 | -1 | linear_all | 0.02 | 2.19291 | 0.01444 | 1.71677 | 0.9853 |
| reference | scalar | 6 | -1 | linear_all | 0.03 | 2.5 | 0.01634 | 1.89185 | 0.9758 |
| reference | scalar | 6 | -1 | linear_all | 0.05 | 2.5 | 0.03166 | 2.15469 | 0.9615 |
| reference | scalar | 6 | -1 | linear_drop_largest | 0.015 | 1.90371 | 0.01747 | 1.53864 | 0.9839 |
| reference | scalar | 6 | -1 | linear_drop_largest | 0.02 | 2.1501 | 0.0162 | 1.65907 | 0.981 |
| reference | scalar | 6 | -1 | linear_drop_largest | 0.03 | 2.5 | 0.01631 | 1.84259 | 0.9712 |
| reference | scalar | 6 | -1 | linear_drop_largest | 0.05 | 2.5 | 0.0316 | 2.11646 | 0.9563 |
| reference | scalar | 6 | -1 | linear_small3 | 0.015 | 1.78606 | 0.03795 | 1.19293 | 0.9468 |
| reference | scalar | 6 | -1 | linear_small3 | 0.02 | 2.10886 | 0.02313 | 1.37535 | 0.9421 |
| reference | scalar | 6 | -1 | linear_small3 | 0.03 | 2.5 | 0.0163 | 1.60996 | 0.9338 |
| reference | scalar | 6 | -1 | linear_small3 | 0.05 | 2.5 | 0.03095 | 1.9312 | 0.923 |
| reference | scalar | 6 | -1 | quadratic_all | 0.015 | 0.977811 | 0.07711 | 0.633288 | 0.7614 |
| reference | scalar | 6 | -1 | quadratic_all | 0.02 | 1.39866 | 0.05979 | 0.826119 | 0.8161 |
| reference | scalar | 6 | -1 | quadratic_all | 0.03 | 2.5 | 0.03018 | 1.1371 | 0.817 |
| reference | scalar | 6 | -1 | quadratic_all | 0.05 | 2.5 | 0.04496 | 1.63461 | 0.805 |
| reference | scalar | 6 | -1 | quadratic_small4 | 0.015 | 0.2 | 0.1773 | -0.150597 | 0.07478 |
| reference | scalar | 6 | -1 | quadratic_small4 | 0.02 | 0.756331 | 0.1576 | 0.257857 | 0.1166 |
| reference | scalar | 6 | -1 | quadratic_small4 | 0.03 | 2.1162 | 0.07126 | 0.639019 | 0.3989 |
| reference | scalar | 6 | -1 | quadratic_small4 | 0.05 | 2.5 | 0.03904 | 1.18468 | 0.5958 |
| reference | scalar | 6 | +1 | linear_all | 0.015 | 1.93474 | 0.01386 | 1.60127 | 0.9889 |
| reference | scalar | 6 | +1 | linear_all | 0.02 | 2.19248 | 0.01444 | 1.71588 | 0.9853 |
| reference | scalar | 6 | +1 | linear_all | 0.03 | 2.5 | 0.01631 | 1.89098 | 0.9758 |
| reference | scalar | 6 | +1 | linear_all | 0.05 | 2.5 | 0.03167 | 2.154 | 0.9615 |
| reference | scalar | 6 | +1 | linear_drop_largest | 0.015 | 1.90341 | 0.01741 | 1.53806 | 0.9837 |
| reference | scalar | 6 | +1 | linear_drop_largest | 0.02 | 2.1497 | 0.01617 | 1.65861 | 0.9809 |
| reference | scalar | 6 | +1 | linear_drop_largest | 0.03 | 2.5 | 0.01625 | 1.84192 | 0.9712 |
| reference | scalar | 6 | +1 | linear_drop_largest | 0.05 | 2.5 | 0.03165 | 2.1161 | 0.9562 |
| reference | scalar | 6 | +1 | linear_small3 | 0.015 | 1.78403 | 0.03775 | 1.19326 | 0.947 |
| reference | scalar | 6 | +1 | linear_small3 | 0.02 | 2.10727 | 0.02305 | 1.37539 | 0.9423 |
| reference | scalar | 6 | +1 | linear_small3 | 0.03 | 2.5 | 0.01619 | 1.60886 | 0.9341 |
| reference | scalar | 6 | +1 | linear_small3 | 0.05 | 2.5 | 0.03101 | 1.93026 | 0.9231 |
| reference | scalar | 6 | +1 | quadratic_all | 0.015 | 0.980854 | 0.07699 | 0.634935 | 0.762 |
| reference | scalar | 6 | +1 | quadratic_all | 0.02 | 1.40285 | 0.05983 | 0.828057 | 0.8165 |
| reference | scalar | 6 | +1 | quadratic_all | 0.03 | 2.5 | 0.03028 | 1.1353 | 0.8187 |
| reference | scalar | 6 | +1 | quadratic_all | 0.05 | 2.5 | 0.04548 | 1.63522 | 0.8047 |
| reference | scalar | 6 | +1 | quadratic_small4 | 0.015 | 0.2 | 0.1777 | -0.151737 | 0.07521 |
| reference | scalar | 6 | +1 | quadratic_small4 | 0.02 | 0.762727 | 0.158 | 0.258132 | 0.116 |
| reference | scalar | 6 | +1 | quadratic_small4 | 0.03 | 2.04478 | 0.07508 | 0.629101 | 0.3963 |
| reference | scalar | 6 | +1 | quadratic_small4 | 0.05 | 2.5 | 0.04019 | 1.17998 | 0.5932 |

## Joint complex surface fits

Model: `y(U,delta)=y00+A U^p+a1 delta[+a2 delta^2+a3 U delta]`.

| config | n | q | variant | Umax | p | rel RMS |
|---|---:|---:|---|---:|---:|---:|
| reference | 4 | -1 | linear_delta | 0.015 | 0.988618 | 0.01596 |
| reference | 4 | -1 | linear_delta | 0.02 | 1.05025 | 0.02036 |
| reference | 4 | -1 | linear_delta | 0.03 | 1.16521 | 0.02732 |
| reference | 4 | -1 | linear_delta | 0.05 | 1.29798 | 0.03856 |
| reference | 4 | -1 | quadratic_cross | 0.015 | 0.2 | 0.002057 |
| reference | 4 | -1 | quadratic_cross | 0.02 | 0.204146 | 0.002275 |
| reference | 4 | -1 | quadratic_cross | 0.03 | 0.2 | 0.00383 |
| reference | 4 | -1 | quadratic_cross | 0.05 | 0.268032 | 0.007514 |
| reference | 4 | -1 | quadratic_delta | 0.015 | 0.988618 | 0.01596 |
| reference | 4 | -1 | quadratic_delta | 0.02 | 1.05025 | 0.02036 |
| reference | 4 | -1 | quadratic_delta | 0.03 | 1.16521 | 0.02732 |
| reference | 4 | -1 | quadratic_delta | 0.05 | 1.29798 | 0.03856 |
| reference | 4 | +1 | linear_delta | 0.015 | 0.988618 | 0.01596 |
| reference | 4 | +1 | linear_delta | 0.02 | 1.05025 | 0.02036 |
| reference | 4 | +1 | linear_delta | 0.03 | 1.16521 | 0.02732 |
| reference | 4 | +1 | linear_delta | 0.05 | 1.29798 | 0.03856 |
| reference | 4 | +1 | quadratic_cross | 0.015 | 0.2 | 0.002057 |
| reference | 4 | +1 | quadratic_cross | 0.02 | 0.204165 | 0.002275 |
| reference | 4 | +1 | quadratic_cross | 0.03 | 0.2 | 0.00383 |
| reference | 4 | +1 | quadratic_cross | 0.05 | 0.268032 | 0.007514 |
| reference | 4 | +1 | quadratic_delta | 0.015 | 0.988618 | 0.01596 |
| reference | 4 | +1 | quadratic_delta | 0.02 | 1.05025 | 0.02036 |
| reference | 4 | +1 | quadratic_delta | 0.03 | 1.16521 | 0.02732 |
| reference | 4 | +1 | quadratic_delta | 0.05 | 1.29798 | 0.03856 |
| reference | 5 | -1 | linear_delta | 0.015 | 1.34238 | 0.03764 |
| reference | 5 | -1 | linear_delta | 0.02 | 1.38479 | 0.03872 |
| reference | 5 | -1 | linear_delta | 0.03 | 1.42556 | 0.04161 |
| reference | 5 | -1 | linear_delta | 0.05 | 1.40494 | 0.05338 |
| reference | 5 | -1 | quadratic_cross | 0.015 | 1.57472 | 0.007271 |
| reference | 5 | -1 | quadratic_cross | 0.02 | 1.62652 | 0.01053 |
| reference | 5 | -1 | quadratic_cross | 0.03 | 1.69198 | 0.01529 |
| reference | 5 | -1 | quadratic_cross | 0.05 | 1.79862 | 0.02212 |
| reference | 5 | -1 | quadratic_delta | 0.015 | 1.34238 | 0.03726 |
| reference | 5 | -1 | quadratic_delta | 0.02 | 1.38479 | 0.03817 |
| reference | 5 | -1 | quadratic_delta | 0.03 | 1.42556 | 0.04094 |
| reference | 5 | -1 | quadratic_delta | 0.05 | 1.40494 | 0.0527 |
| reference | 5 | +1 | linear_delta | 0.015 | 1.34239 | 0.03764 |
| reference | 5 | +1 | linear_delta | 0.02 | 1.38479 | 0.03872 |
| reference | 5 | +1 | linear_delta | 0.03 | 1.42556 | 0.04161 |
| reference | 5 | +1 | linear_delta | 0.05 | 1.40494 | 0.05338 |
| reference | 5 | +1 | quadratic_cross | 0.015 | 1.57473 | 0.007271 |
| reference | 5 | +1 | quadratic_cross | 0.02 | 1.62652 | 0.01053 |
| reference | 5 | +1 | quadratic_cross | 0.03 | 1.69198 | 0.01529 |
| reference | 5 | +1 | quadratic_cross | 0.05 | 1.79862 | 0.02212 |
| reference | 5 | +1 | quadratic_delta | 0.015 | 1.34239 | 0.03726 |
| reference | 5 | +1 | quadratic_delta | 0.02 | 1.38479 | 0.03817 |
| reference | 5 | +1 | quadratic_delta | 0.03 | 1.42556 | 0.04094 |
| reference | 5 | +1 | quadratic_delta | 0.05 | 1.40494 | 0.0527 |
| reference | 6 | -1 | linear_delta | 0.015 | 1.33112 | 0.03708 |
| reference | 6 | -1 | linear_delta | 0.02 | 1.45296 | 0.04531 |
| reference | 6 | -1 | linear_delta | 0.03 | 1.72516 | 0.0546 |
| reference | 6 | -1 | linear_delta | 0.05 | 2.16465 | 0.06381 |
| reference | 6 | -1 | quadratic_cross | 0.015 | 0.2 | 0.007301 |
| reference | 6 | -1 | quadratic_cross | 0.02 | 0.2 | 0.01099 |
| reference | 6 | -1 | quadratic_cross | 0.03 | 0.2 | 0.01972 |
| reference | 6 | -1 | quadratic_cross | 0.05 | 0.223659 | 0.03267 |
| reference | 6 | -1 | quadratic_delta | 0.015 | 1.33112 | 0.03708 |
| reference | 6 | -1 | quadratic_delta | 0.02 | 1.45296 | 0.04531 |
| reference | 6 | -1 | quadratic_delta | 0.03 | 1.72516 | 0.0546 |
| reference | 6 | -1 | quadratic_delta | 0.05 | 2.16465 | 0.0638 |
| reference | 6 | +1 | linear_delta | 0.015 | 1.33111 | 0.03708 |
| reference | 6 | +1 | linear_delta | 0.02 | 1.45296 | 0.04531 |
| reference | 6 | +1 | linear_delta | 0.03 | 1.72517 | 0.0546 |
| reference | 6 | +1 | linear_delta | 0.05 | 2.16464 | 0.06381 |
| reference | 6 | +1 | quadratic_cross | 0.015 | 0.2 | 0.007301 |
| reference | 6 | +1 | quadratic_cross | 0.02 | 0.2 | 0.01099 |
| reference | 6 | +1 | quadratic_cross | 0.03 | 0.2 | 0.01972 |
| reference | 6 | +1 | quadratic_cross | 0.05 | 0.223658 | 0.03267 |
| reference | 6 | +1 | quadratic_delta | 0.015 | 1.33111 | 0.03708 |
| reference | 6 | +1 | quadratic_delta | 0.02 | 1.45296 | 0.04531 |
| reference | 6 | +1 | quadratic_delta | 0.03 | 1.72517 | 0.05459 |
| reference | 6 | +1 | quadratic_delta | 0.05 | 2.16464 | 0.0638 |

## Interpretation

1. Jordan survival requires high alignment, a small trace fraction, and a small `|Bx-iBz|/(|Bx|+|Bz|)` mismatch in reliable data.
2. Generic second-order EP perturbation requires stable `p approximately 1` across detuning models, U windows, NRG iterations, and charge sectors.
3. The stronger quartic relation requires the complex coefficient `C=Q/(U^2 beta0^2 F)` to approach a nonzero plateau with stable magnitude and phase.
4. `C_gamma` is exported separately because this adapter distinguishes the raw control `beta0` from the physical local matrix element `Gamma_PT`.
5. The current SOC-overlap option remains an effective control model, not the microscopic `k^2 +/- lambda k` bath.

## Output tables

- `tracked_pairs_all_iterations.csv`: branch-resolved pole and residue data.
- `zero_detuning_ensemble.csv`: all zero-detuning extrapolation methods.
- `exponent_stability.csv`: U-window, method, iteration, and sector dependence.
- `joint_surface_fits.csv`: simultaneous complex U/detuning fits.
- `quartic_plateau_metrics.csv`: complex plateau tests.
- `iteration_stability.csv`, `sector_symmetry.csv`, `convergence_summary.csv`: robustness checks.
