module mlC;

// ML stuff which doesn't fit anywhere else

import std.math;

class RngA {
    double v = 0.01;

    // returns in range 0.0;1.0
    final double nextReal() {
        v += 0.01;
        return (1.0 + cos(v*10000000000.0)) * 0.5;
    }
	
	final long nextInteger(long max) {
		return cast(long)(nextReal() * max);
	}
}

double[] genRngVec(long size, RngA rng) {
    double[] res;
    for(long it=0; it<size; it++) {
        res ~= (rng.nextReal()*2.0 - 1.0);
    }
    return res;
}

