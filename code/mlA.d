// module to encapsulate basic ML stuff
module mlA;


import std.math;

class Vec {
    public double[] arr;

    public this(double[] arr) {
        this.arr = arr;
    }

    public this() {
    }

    public Vec opBinary(string op : "+")(Vec rhs) {
        double[] res;
        for (long idx=0;idx<arr.length;idx++) {
            res ~= (arr[idx] + rhs.arr[idx]);
        }
        return new Vec(res);
    }

    public static Vec scale(Vec v, double s) {
        Vec res = new Vec();
        foreach (iv; v.arr) {
            res.arr ~= (iv * s);
        }
        return res;
    }
}

double dot(Vec a, Vec b) {
    double res = 0.0;
    for(long idx=0;idx<a.arr.length;idx++) {
        res += (a.arr[idx]*b.arr[idx]);
    }
    return res;
}

Vec normalize(Vec v) {
    double l = sqrt(dot(v, v));
    return Vec.scale(v, 1.0/l);
}

double calcL2norm(Vec arg) {
	return sqrt(dot(arg, arg));
}

double calcCosineSim(Vec a, Vec b) {
	return dot(a, b) / (calcL2norm(a)*calcL2norm(b));
}

Vec mulComponents(Vec a, Vec b) {
	double[] arrRes = [];
	for(long idx=0;idx<a.arr.length;idx++) {
        arrRes ~= (a.arr[idx]*b.arr[idx]);
    }
	return new Vec(arrRes);
}

Vec makeVecByLength(long length) {
    double[] arr;
    for(long count=0;count<length;count++) {
        arr ~= 0.0;
    }
    return new Vec(arr);
}

int calcHighestValueIdx(Vec arg) {
    int highestIdx = 0;
    double highestVal = arg.arr[0];
    for(int idx=0;idx<arg.arr.length;idx++) {
        if (arg.arr[idx] > highestVal) {
            highestVal = arg.arr[idx];
            highestIdx = idx;
        }
    }
    return highestIdx;
}

Vec append(Vec a, Vec b) {
    double[] arr;
    foreach (iv; a.arr) {
        arr ~= iv;
    }
    foreach (iv; b.arr) {
        arr ~= iv;
    }
    return new Vec(arr);
}

Vec add(Vec a, Vec b) {
	double[] arr;
	for (long i=0; i<a.arr.length; i++) {
		arr ~= (a.arr[i] + b.arr[i]);
	}
	return new Vec(arr);
}

Vec scale(Vec v, double s) {
	double[] arr;
    foreach (iv; v.arr) {
        arr ~= (iv*s);
    }
	return new Vec(arr);
}

Vec vecMake(double v, long size) {
	double[] arr;
	for (long i=0; i<size; i++) {
		arr ~= v;
	}
	return new Vec(arr);
}

Vec oneHotEncode(long symbol, long size) {
    double[] arr;
    for(int i=0;i<size;i++) {
        arr ~= 0.0;
    }
    arr[symbol] = 1.0;
    return new Vec(arr);
}

/* commented because not needed
string convVecToString(Vec v) {
    string res = "";
    for (long idx=0; idx < v.arr.length; idx++) {
        res ~= format("%f", v.arr[idx]);
        if (idx < v.arr.length-1) {
            res ~= ",";
        }
    }
    return res;
}

string convVecToPythonString(Vec v) {
    return "["~convVecToString(v)~"]";
}
*/









class DatItem {
    public Vec v;
    public string guid; // unique id identifying the item

    public long class_;

    final this(Vec v, string guid, long class_) {
        this.v = v;
        this.guid = guid;
        this.class_ = class_;
    }
}

// Prototype based classifier
public class ClassifierA {

    public DatItem[] items;


    public Vec inference(Vec v, bool learn=false) {
        // TODO : implement learning!

        double[] dotResults = [];
        foreach (iItem; items) {
            double dotResult = dot(v, iItem.v);
            dotResults ~= dotResult;
        }

        Vec normalized = normalize(new Vec(dotResults));

        // compute scaled result vector
        Vec accu = makeVecByLength(items[0].v.arr.length);
        foreach (iIdx, iDot; dotResults) {
            accu = accu + Vec.scale(items[iIdx].v, iDot);
        }

        return accu;
    }

}

