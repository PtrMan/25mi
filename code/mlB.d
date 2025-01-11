// module for ML

// implements basic stuff for gradient descent

module mlB;

import std.stdio : writeln;
//import std.string : join;
import std.format : format;

import std.math : exp;

class Dual {
    public double real_; // value
    public double grad; // differential

    public final this(double real_, double grad) {
        this.real_ = real_;
        this.grad = grad;
    }
}

Dual add(Dual lhs, Dual rhs) {
    return new Dual(rhs.real_+lhs.real_, rhs.grad+lhs.grad);
}

Dual sub(Dual lhs, Dual rhs) {
    return new Dual(rhs.real_-lhs.real_, rhs.grad-lhs.grad);
}

Dual mul(Dual lhs, Dual rhs) {
    double real_ = lhs.real_*rhs.real_;
    double grad = lhs.real_*rhs.grad + lhs.grad*rhs.real_;
    return new Dual(real_, grad);
}

Dual exp(Dual arg) {
    double real_ = exp(arg.real_);
    double grad = arg.grad*real_;
    return new Dual(real_, grad);
}

Dual activationFunction_leakyRelu(Dual arg) {
    if (arg.real_ >= 0.0) {
        return arg;
    }
    else {
        return mul(arg, new Dual(-0.01, 0.0));
    }
}

class DualVec {
    Dual[] arr;

    final this(Dual[] arr) {
        this.arr = arr;
    }
}

DualVec add(DualVec a, DualVec b) {
    Dual[] res;
    for(long idx=0;idx<a.arr.length;idx++) {
        res ~= add(a.arr[idx], b.arr[idx]);
    }
    return new DualVec(res);
}

DualVec sub(DualVec a, DualVec b) {
    Dual[] res;
    for(long idx=0;idx<a.arr.length;idx++) {
        res ~= sub(a.arr[idx], b.arr[idx]);
    }
    return new DualVec(res);
}

DualVec mul(DualVec a, DualVec b) {
    Dual[] res;
    for(long idx=0;idx<a.arr.length;idx++) {
        res ~= mul(a.arr[idx], b.arr[idx]);
    }
    return new DualVec(res);
}

DualVec applyFunction(DualVec arg, Dual function(Dual) fn) {
    Dual[] arrRes;
    foreach(iv; arg.arr) {
        arrRes ~= fn(iv);
    }
    return new DualVec(arrRes);
}

Dual dot(DualVec a, DualVec b) {
	Dual arrRes = new Dual(0.0, 0.0);
	for(long idx=0;idx<a.arr.length;idx++) {
		arrRes = add(mul(a.arr[idx], b.arr[idx]), arrRes);
	}
	return arrRes;
}





// unit of NN with automatic differentiation
class UnitA {
	DualVec weights;
	Dual bias;
	Dual function(Dual) activationFn;

	final this(DualVec weights, Dual bias, Dual function(Dual) activationFn) {
		this.weights = weights;
		this.bias = bias;
		this.activationFn = activationFn;
	}
}

void QQQ_NN029982()
{

	double lr = 0.05;



	UnitA[] units;
	
	
	
	
	// unit for testing
	{
		UnitA createdUnit = new UnitA(new DualVec([new Dual(-0.38382, 0.0), new Dual(0.19033401, 0.0), new Dual(0.2003, 0.0)]), new Dual(0.00001, 0.0), &activationFunction_leakyRelu);
		units ~= createdUnit;
	}
	
	long selUnitIdx = 0;
	long selParameterIdx = 1; // negative is selection of bias
	
	units[selUnitIdx].weights.arr[selParameterIdx].grad = 0.0;
	
	selUnitIdx = 0;
	selParameterIdx = 1;
	
	units[selUnitIdx].weights.arr[selParameterIdx].grad = 1.0;
	
	for(long itTraining=0; itTraining<1000; itTraining++) {
		
		DualVec in0 = new DualVec([new Dual(0.9, 0.0), new Dual(-0.9, 0.0), new Dual(-0.9, 0.0)]);

		// compute result of layer

		Dual[] outputArr;
		foreach (itUnit; units) {
			Dual activation = dot(in0, itUnit.weights);
			Dual activation2 = add(activation, itUnit.bias);
			Dual output = itUnit.activationFn(activation2);
			outputArr ~= output;
		}
		DualVec output = new DualVec(outputArr);


		// debug outputArr
		if (false) {
			writeln("");
			foreach (itVal; output.arr) {
				writeln(format("%f %f", itVal.real_, itVal.grad));
			}
		}
		
		
		DualVec yTraining = new DualVec([new Dual(0.9, 0.0)]);
		
		
		DualVec diff = sub(output, yTraining);
		
		// compute squared sum
		Dual sum_ = new Dual(0.0, 0.0);
		foreach (itVal; diff.arr) {
			sum_ = add(sum_, mul(itVal, itVal));
		}
		
		if ((itTraining % 150) == 0) {
			writeln(format("%f %f", sum_.real_, sum_.grad));
		}
		


		
		//units[0].bias.real_ -= (lr * sum_.grad);
		//units[0].weights.arr[0].real_ -= (lr * sum_.real_);
		
		if (selParameterIdx >= 0) {
			units[selUnitIdx].weights.arr[selParameterIdx].real_ -= (lr * sum_.real_);
		}
		else {
			units[selUnitIdx].bias.real_ -= (lr * sum_.real_);
		}
	}
}


