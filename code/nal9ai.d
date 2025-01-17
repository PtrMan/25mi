
// compile with
//
//  copy C:\Users\rober\fsRoot\coding\github25mi\25mi\code\*.d . ; dmd nal9ai.d mlA.d mlB.d mlC.d aeraish.d; .\nal9ai.exe

import std.stdio;
import std.string : join;
import std.format : format;

// import std.datetime.stopwatch : StopWatch, AutoStart;


import std.math : cos;





// for GUID generation
import std.uuid;



import core.stdc.stdlib : exit;




import mlA;

// for testing
import mlB;

import mlC;

import aeraish;

void main() {



	// manual test: association machine
	if (false) {
		manualtest__softComputingAssocA();
	}
	
	
	// unitest for ARC solver based on association machine
	{
		IDEA_LAB__drawingTaskSimpleA();
	}

    writeln("FIN");
}

































// highly experimental area

// basic idea: 
// we have units
//    traits of units: program unit's with traits, such as need to survive by competing for resources (for example up-votes when a stimuli matches to the pattern detector its looking out for). 
//
// all is orchestrated by a "soup manager" - which determines the overall task of the units

class UnitB {
	Vec v; // vector to look out
	
	string guid; // unique id which identifies this unit
	
	UnitEvidence unitEvidence = new UnitEvidence();
	
	
	// attribute for stimuli + action mapping
	string actionCode; // associated action code
	Vec consequenceVec; // vector of the consequence
	int predictedReward = 0; // predicted reward which is associated with the consequence after the action
	
	
	Vec attentionMask; // attention mask for which this unit is looking out for
	
	
	this(string guid) {
		this.guid = guid;
	}
}

string retDebugStrOfUnit(UnitB unit) {
	return format("v         =%s\nactionCode=%s\npredictedReward=%s\n\n", unit.v.arr, unit.actionCode, unit.predictedReward);
}


// evidence for the unit
class UnitEvidence {
	long positiveMatchCnt = 0; // matching counter for positive matches
	
	// TODO ::: Time lastPositiveMatchTime = null; // time of last positive matching

	this() {
	}
	
	final void addPositive() {
		positiveMatchCnt += 1;
	}
}


// context which contains units
class CtxZZZ {
	UnitB[] units;
}

class ZZZx {
	// context we are using
	CtxZZZ ctx = new CtxZZZ();

	final void identifyAndLearn(Vec v) {
		double bestSim = -1.0;
		UnitB bestUnit = null;
	
		foreach (itUnit; ctx.units) {
			double sim = calcCosineSim(v, itUnit.v);
			if (sim > bestSim) {
				bestSim = sim;
				bestUnit = itUnit;
			}
		}
		
		double thresholdSim = 0.85;
		if (bestSim > thresholdSim) {
			// reward winner unit
			bestUnit.unitEvidence.addPositive();
		}
		else {
			// we add a new unit
			// TODO LOW : generate GUID
			UnitB createdUnit = new UnitB("");
			
			// reward created unit
			createdUnit.unitEvidence.addPositive();
			
			
			ctx.units ~= createdUnit;
			// TODO : care about AIKR here
		}
	}
}





// Jeff Hawkin : "reference frames"  : explained in the book "The 1000 brains theory"











// strategy to calculate similarity
abstract class SimilarityCalculationStrategy {
	abstract double[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx);
}

// soft computing similarity
class SoftMaxSimilarityCalculationStrategy : SimilarityCalculationStrategy {
	override double[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx) {
		double[] arrSim = [];
	
		foreach (itIdx, itUnit; ctx.units) {
			double sim = calcCosineSim(stimulus, itUnit.v);
			double sim2 = (sim+1.0) * 0.5; // map to 0.0 1.0 range
			arrSim ~= sim2;
		}
		
		return arrSim;
	}
}

// hard similarity
class HardMaxSimilarityCalculationStrategy : SimilarityCalculationStrategy {
	override double[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx) {
		double bestSim = -1.0;
		long bestUnitIdx = -1;
	
		foreach (itIdx, itUnit; ctx.units) {
			double sim = calcCosineSim(stimulus, itUnit.v);
			if (sim > bestSim) {
				bestSim = sim;
				bestUnitIdx = itIdx;
			}
		}
		
		double[] arrSim = [];
		foreach (itIdx, itUnit; ctx.units) {
			arrSim ~= 0.0;
		}
		arrSim[bestUnitIdx] = (bestSim+1.0) * 0.5;
		
		return arrSim;
	}
}


















// temporarily holds the result of the planning
class AlgorithmResult__Planning {
	string firstActionActionCode; // actionCode of the first action which the planning algorithm has computed
	double expectedRewardSum = 0.0; // exptected reward sum for the selected action
}

// use of predicted input for 
// mechanism:
// a) feed input X to column to predict input X which follows, together with the action and reward
// b) goto a)

// TODO IDEA: we could predict the next input and the reward with NN which are trained.

AlgorithmResult__Planning LAB__cortialAlgorithm__planning_A(Vec stimulus, ColumnCtxA columnCtx, int nPlanningDepth) {



	//int nPlanningDepth = 1; // how many iterations are done for planning
	//int nPlanningDepth = 4; // how many iterations are done for planning
	

	Vec iteratedStimulus = stimulus;
	
	double expectedRewardSum = 0.0; // sum of rewards of the "path"
	string firstActionActionCode = null;
	
	if (columnCtx.ctx.units.length > 0) { // there must be units to vote on
		for(int itPlanningDepth=0;itPlanningDepth<nPlanningDepth;itPlanningDepth++) {
			// vote for best unit
			VotingWeightsOfUnits votingWeights = iteratedPlanning__voteUnitsAsVotingWeights(iteratedStimulus, columnCtx);
			
			// select winner unit weights
			VotingWeightsOfUnits votingWeightsAfterSelectingWinner = iteratedPlanning__selectWinnerUnitVector(votingWeights);
			
			if (itPlanningDepth == 0) {
				firstActionActionCode = columnCtx.ctx.units[ calcIndexWithHighestValue(votingWeightsAfterSelectingWinner) ].actionCode;
			}
			
			expectedRewardSum += (calcWeightedPredictedReward(votingWeightsAfterSelectingWinner.v, columnCtx) * exp(-cast(double)itPlanningDepth * 0.9));
			
			// compute prediction of predicted output by vector
			Vec vecPredicted = computePredictedVector(votingWeightsAfterSelectingWinner, columnCtx);
			
			iteratedStimulus = vecPredicted; // feed as stimulus for next iteration
		}
	}
	
	
	// now we have a action "firstActionActionCode" which leads to a possible path with expected reward = "expectedRewardSum"
	
	// we have to do this a few times and select the action which gives us the highest expected reward
	
	// TODO : implement outer loop
	
	AlgorithmResult__Planning res = new AlgorithmResult__Planning();
	res.firstActionActionCode = firstActionActionCode;
	res.expectedRewardSum = expectedRewardSum;
	
	return res;
}


// typed helper class to give a vector which is the normalized weight a type
class VotingWeightsOfUnits {
	Vec v;
	
	final this(Vec v) {
		this.v = v;
	}
}

VotingWeightsOfUnits iteratedPlanning__voteUnitsAsVotingWeights(Vec stimulus, ColumnCtxA columnCtx) {
	
	SimilarityCalculationStrategy similarityCalcStrategy;
	similarityCalcStrategy = new SoftMaxSimilarityCalculationStrategy();
	//similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy(); // use attention strategy
	double[] arrSim = similarityCalcStrategy.calcMatchingScore__by__stimulus(stimulus, columnCtx.ctx);
	
	return new VotingWeightsOfUnits( normalize( new Vec(arrSim) ) );
}


VotingWeightsOfUnits iteratedPlanning__selectWinnerUnitVector(VotingWeightsOfUnits votingWeights) {
	
	long maxIdx = 0;
	double maxValue = -double.max;
	foreach (itIdx, itVal; votingWeights.v.arr) {
		if (itVal > maxValue) {
			maxValue = itVal;
			maxIdx = itIdx;
		}
	}
	
	Vec vecOneHot = oneHotEncode(maxIdx, votingWeights.v.arr.length);
	
	return new VotingWeightsOfUnits(vecOneHot);
}

double calcWeightedPredictedReward(Vec v, ColumnCtxA columnCtx) {
	double res = 0.0;
	for (int idx=0; idx<v.arr.length; idx++) {
		res += ( v.arr[idx] * columnCtx.ctx.units[idx].predictedReward );
	}
	return res;
}

Vec computePredictedVector(VotingWeightsOfUnits votingWeights, ColumnCtxA columnCtx) {
	Vec res = makeVecByLength(columnCtx.ctx.units[0].consequenceVec.arr.length);
	
	for (int idxUnit=0; idxUnit<columnCtx.ctx.units.length; idxUnit++) {
		res = add( scale(columnCtx.ctx.units[idxUnit].consequenceVec, votingWeights.v.arr[idxUnit]), res);
	}
	
	return res;
}



int calcIndexWithHighestValue(VotingWeightsOfUnits votingWeights) {
	return calcHighestValueIdx(votingWeights.v);
}

















// TODO : implement soft mapper of
// observed state + action -> effect state
//
// with using SoftMaxSimilarityCalculationStrategy to compute the blend of the interpolation to vote on best next action

// TODO TODO TOOD TODO






// context which maps to a single crtial column
class ColumnCtxA {
	CtxZZZ ctx = new CtxZZZ();
	
	string[] availableActions;
	
	Vec lastPerceivedStimulus = null;
	string lastSelectedAction = null;
	
	//  0 is no reward signal
	//  1 is positive reward
	// -1 is negative reward
	int lastRewardFromEnvironment = 0;
}

/*

learner is based on following ideas:

* cortial algorithms as core methodology of substrate

* computing similarity between neurons :  modern hopfield neural networds
* unsupervised learning of prediction  :  hierachical temporal memory (HTM) theory

* TODO : decision making by voting of multiple column: TODO : searhc paper from hawkins

*/

// context of the cortial algorithm - learning and inference and interaction with the environment
class CortialAlgoithm_LearnerCtx {
	ColumnCtxA column = new ColumnCtxA();
	
	
	// reward statistics
	long cntRewardPos = 0;
	long cntRewardNeg = 0;
	
	
	double paramRandomActionChance = 0.1; // probability to act with a random action in each timestep
	
	double paramGoodEnoughSimThreshold = 0.95;
	

	RngA rng = new RngA();
	
	
	EnvAbstract env = null; // must be set externally
	
	
	
	void resetColumnStates() {
		
		column.lastPerceivedStimulus = null;
		column.lastSelectedAction = null;
	
	}
	
	
	// synchronous step between learner and environment
	void learnerSyncronousAndEnviromentStep(long globalIterationCounter) {
		
		writeln("");
		writeln("");
		writeln("");
		
		// DEBUG units
		{
			writeln("units:");
			foreach (itUnit; column.ctx.units) {
				writeln("");
				writeln(retDebugStrOfUnit(itUnit));
			}
			writeln("");
		}
		
		
		env.setGlobalIterationCounter(globalIterationCounter);
		
		Vec perceivedStimulus;
		
		perceivedStimulus = env.receivePerception();
		
		writeln(format("perceived stimulus= %s", perceivedStimulus.arr));
		
		
		
		// build contigency from past observation, past action and current stimulus and update in memory
		buildContingencyFromPastObservation(perceivedStimulus);
		
		
		
		// STAGE: compare stimulus to prototypes and compute for each unit: similarity of stimulus to compared patttern of unit
		
		// use calc__action__byVotingMax to decide on the next action based on the observed stimulus state . then update the associations based on the observed effect state
		
		
		
		SimilarityCalculationStrategy similarityCalcStrategy;
		similarityCalcStrategy = new SoftMaxSimilarityCalculationStrategy();
		//similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy(); // use attention strategy
		double[] arrSim = similarityCalcStrategy.calcMatchingScore__by__stimulus(perceivedStimulus, column.ctx);
		
		// DEBUG
		writeln("");
		writeln("similarities to perceivedStimulus:");
		writeln(arrSim);
		
		
		
		
		/* commented because DEPRECTAED because new functionality can do this much better!
		
		// we need to mask units which learned negative reward so that we don't pick actions which lead to negative reward
		double[] arrPredictedReward = [];
		for (long itIdx=0; itIdx<arrSim.length; itIdx++) {
			//if (column.ctx.units[itIdx].predictedReward < 0) { // check if is enough negative reward
				//arrSim[itIdx] = 0.0; // mask similarity so it wont get selected as next action
				arrPredictedReward ~= column.ctx.units[itIdx].predictedReward;
			//}
		}
		
		
		

		// compute the action which has the highest votes
		// 
		// result is the actionCode with the highest number of votes
		string calc__action__byVotingMax(CtxZZZ ctx, double[] arrSim, double[] arrPredictedReward) {
			double[string] voteStrengthByAction;
			
			foreach (itIdx, itUnit; ctx.units) {
				string associatedActionOfUnit = itUnit.actionCode;
				
				if (!(associatedActionOfUnit in voteStrengthByAction)) {
					voteStrengthByAction[associatedActionOfUnit] = 0.0;
				}
				
				if (arrPredictedReward[itIdx] > 0) {
					
					int strategyForSimAndRewardFusion = 0; // strategy for fusion of similarity of stimulus and prototype and reward
					
					if (strategyForSimAndRewardFusion == 0) {
						voteStrengthByAction[associatedActionOfUnit] += arrSim[itIdx];
					}
					else {
						voteStrengthByAction[associatedActionOfUnit] += ( arrSim[itIdx] * (cast(double)arrPredictedReward[itIdx] + 0.01) );
					}
				}
			}
			
			// search max
			double maxVotingVal = -double.max;
			string maxVotingActionCode = null;
			foreach (itKey, itValue; voteStrengthByAction) {
				if (voteStrengthByAction[itKey] > maxVotingVal) {
					maxVotingVal = voteStrengthByAction[itKey];
					maxVotingActionCode = itKey;
				}
			}
			
			return maxVotingActionCode;
		}
		
		string maxVotingActionCode = calc__action__byVotingMax(column.ctx, arrSim, arrPredictedReward);
		
		*/
		
		int nPlanningDepth = 2;
		AlgorithmResult__Planning resPlanning = LAB__cortialAlgorithm__planning_A(perceivedStimulus, column, nPlanningDepth);
		
		string maxVotingActionCode = resPlanning.firstActionActionCode; // copy selected action code into temporary variable
		
		
		
		
		
		// DEBUG
		writeln(format("selected max firstActionActionCode=%s    expected future rewardSum=%f", resPlanning.firstActionActionCode, resPlanning.expectedRewardSum));
		
		string selectedActionCode = null;
		
		if (maxVotingActionCode !is null) {
			selectedActionCode = maxVotingActionCode;
		}
		
		
		
		
		/*
		
		double maxVal = -1.0;
		long idxMax = -1;
		foreach (itIdx, itVal; arrSim) {
			if (itVal > maxVal) {
				maxVal = itVal;
				idxMax = itIdx;
			}
		}
		
		
		writeln(format("similarityMaxVal=%f", maxVal));
		*/
		
		
		// stage: reward winner unit and punish looser units
		
		// now we reward the winning unit and punish the loosing unit (only do this when the column is using attention)
		{
			// TODO LOW			
		}
		
		
		// idea which doesnt look good: create new unit when max(arrSim) is below a threshold. this is also incompatible with the attention idea
		{
			/*
			if (maxVal < 0.75) { // no winner unit was found!
				//
				
				UnitB createdUnit = new UnitB("");
				createdUnit.v = perceivedStimulus;
				
				// reward created unit
				createdUnit.unitEvidence.addPositive();
				
				
				ctx.units ~= createdUnit;
				// TODO : care about AIKR here		
				
			}
			else { // winner unit was found
				
				// reward winner unit
				
			}
			*/
		}


		
		// stage: RANDOM ACTION
		
		// select random action 
		bool enSelRandomActionThisStep = maxVotingActionCode is null || rng.nextReal() < paramRandomActionChance; // do we select a random action at this step?
		if (enSelRandomActionThisStep) {
			long idxSel = rng.nextInteger(column.availableActions.length);
			selectedActionCode = column.availableActions[idxSel];
			writeln("DBG  random action is selected");
		}
		
		
		// DEBUG
		writeln(format("selected actionCode=%s", selectedActionCode));
		
		// stage: DO ACTUAL ACTION
		{
			env.doAction(selectedActionCode);
		}
		
		
		
		// stage: receive reward from environment
		{
			// NOTE that 0 is no reward signal
			column.lastRewardFromEnvironment = env.retRewardFromLastAction();
			
			if (column.lastRewardFromEnvironment < 0) {
				cntRewardNeg+=1;
			}
			else if (column.lastRewardFromEnvironment > 0) {
				cntRewardPos+=1;
			}
			
			if (column.lastRewardFromEnvironment < 0) {
				writeln("column: received - NEGATIVE reward!");
			}
			else if (column.lastRewardFromEnvironment > 0) {
				writeln("column: received + POSITIVE reward!");
			}
			
			// TODO : use reward before next cycle to reward/punish units
		}
		
		
		// stage: prepare next cycle
		column.lastSelectedAction = selectedActionCode;
		column.lastPerceivedStimulus = perceivedStimulus;
		
		
		
		
		// statistics: we output here statistics about the reward thus far
		{
			writeln("");
			double rationOfReward = cast(double)cntRewardPos / cast(double)(cntRewardPos + cntRewardNeg);
			writeln(format("global: ratio of reward=%f", rationOfReward));
		}
	}
	
	// give the learner a chance to learn from the last step
	void finish() {
		Vec perceivedStimulus;
		
		perceivedStimulus = env.receivePerception();
		
		writeln(format("perceived stimulus= %s", perceivedStimulus.arr));
		
		
		
		// build contigency from past observation, past action and current stimulus and update in memory
		buildContingencyFromPastObservation(perceivedStimulus);
	}
	
	
	
	
	
	
	// (private)
	//
	// build contigency from past observation, past action and current stimulus and update in memory
	//
	void buildContingencyFromPastObservation(Vec perceivedStimulus) {
		// NOTE : we only update if it is similar enough to existing condition
		
		if (column.lastPerceivedStimulus !is null) {
			
			bool wasMatchFound = false;
			
			// search for match
			SimilarityCalculationStrategy similarityCalcStrategy = new SoftMaxSimilarityCalculationStrategy();
			double[] arrSim = similarityCalcStrategy.calcMatchingScore__by__stimulus(column.lastPerceivedStimulus, column.ctx);
			
			// DEBUG
			writeln("");
			writeln("sim to perceived contigency:");
			writeln(arrSim);
			
			
			
			// decide if the match is good enough
			{
				long idxBest = -1;
				double valBest = -2.0;
				for (long itIdx=0; itIdx<arrSim.length; itIdx++) {
					if (column.lastSelectedAction == column.ctx.units[itIdx].actionCode) { // action must be the same to count as the same
						if (arrSim[itIdx] > valBest) {
							idxBest = itIdx;
							valBest = arrSim[itIdx];
						}
					}
				}
				
				if (valBest > paramGoodEnoughSimThreshold) {
					
					// make sure that the action is the same. I guess this is very important that actions which dont get reward by matching die off at some point.
					if (column.ctx.units[idxBest].actionCode == column.lastSelectedAction) {
						wasMatchFound = true;
						
						// reward unit
						// TODO LOW
					}
				}
			}
			
			if (!wasMatchFound) {
				// we add a new unit if no match was found
				
				// we add a new unit
				UnitB createdUnit = new UnitB(randomUUID().toString());
				createdUnit.v = column.lastPerceivedStimulus;
				createdUnit.attentionMask = vecMake(1.0, createdUnit.v.arr.length); // attention mask which doesn't change any channels by default (to make testing easier)
				createdUnit.actionCode = column.lastSelectedAction;
				createdUnit.consequenceVec = perceivedStimulus;
				
				createdUnit.predictedReward = column.lastRewardFromEnvironment; // attach reward to be able to predict reward
				// TODO LOW : revise predictedReward via some clever formula when a good enough unit was found
				
				// reward created unit
				createdUnit.unitEvidence.addPositive();
				
				
				column.ctx.units ~= createdUnit;
				// TODO : care about AIKR here
				
				writeln("");
				writeln("DBG :  created new unit by contingency");
			}
		}
	}
}




void manualtest__softComputingAssocA() {

	
	CortialAlgoithm_LearnerCtx learner = new CortialAlgoithm_LearnerCtx();
	
	
	//learner.env	= new PerceptualDummy0Env(); // create dummy environment
	learner.env = new Simple0Env(); // create simple environment for unittesting
	
	learner.env = new SimpleAssoc0Env(); // simple association environment for 
	
	
	
	learner.column.availableActions = [];
	learner.column.availableActions ~= "^a";
	learner.column.availableActions ~= "^b";
	learner.column.availableActions ~= "^c";
	
	
	learner.resetColumnStates();
	
	
	for (long globalIterationCounter=0;globalIterationCounter<2000;globalIterationCounter++) {
		learner.learnerSyncronousAndEnviromentStep(globalIterationCounter);
		globalIterationCounter += 1;		
	}
	

}







class EnvAbstract {
	// perceive the perception from the environment
	abstract Vec receivePerception();

	// do action in the environment
	abstract void doAction(string selectedActionCode);
	
	// return  0 : no signal
	// return  1 : if positive reward
	// return -1 : if negative reward
	abstract int retRewardFromLastAction();
	
	
	
	// helper to inform environment about current global iteration counter
	abstract void setGlobalIterationCounter(long iterationCnt);
}

class PerceptualDummy0Env : EnvAbstract {
	long iterationCnt = 0; // (private) iteration counter : used to generate different dummy stimulus from the environment

	this() {
	}
	
	override Vec receivePerception() {
		Map2d map0 = new Map2d(Vec2i(3, 3));
		map0.writeAt(1, Vec2i(iterationCnt % map0.retSize().x, 2));
		
		Vec perceivedStimulus = conv_map2d_to_arrOneHot(map0, 12);
		return perceivedStimulus;
	}
	
	override void doAction(string selectedActionCode) {		
		// do nothing
	}
	
	override int retRewardFromLastAction() {
		return 0;
	}
	
	override void setGlobalIterationCounter(long iterationCnt) {
		this.iterationCnt = iterationCnt;
	}
}


// very simple environment which rewards only one action and punishes all others
//
// is used to UNITTEST the core learner that it learns to pick the correct action
class Simple0Env : EnvAbstract {
	private string lastAction = null;
	
	string correctAction = "^b";

	this() {
	}
	
	override Vec receivePerception() {
		return new Vec([1.0]); // return dummy perception
	}
	
	override void doAction(string selectedActionCode) {		
		lastAction = selectedActionCode; // remember action to compute reward
	}
	
	// return  0 : no signal
	// return  1 : if positive reward
	// return -1 : if negative reward
	override int retRewardFromLastAction() {
		if (lastAction == correctAction) {
			return 1;
		}
		return -1;
	}
	
	override void setGlobalIterationCounter(long iterationCnt) {
	}
}


// simple environment which tests if a association of a color to a action is usccessfully learned by the learner
class SimpleAssoc0Env : EnvAbstract {
	
	string correctAction = null; // private
	
	RngA rngEnv = new RngA();
	
	private string lastAction = null;
	
	this() {	
	}
	
	
	override Vec receivePerception() {
		Vec vecOut = vecMake(0.001, 3);
		
		string[] actions = ["^a", "^b", "^c"];
		
		long selIdx = rngEnv.nextInteger(actions.length);
		vecOut.arr[selIdx] = 1.0; // encode perception so learner can associate stimuli with correct action
		correctAction = actions[selIdx]; // remember correct action
		
		return vecOut;
	}
	
	override void doAction(string selectedActionCode) {		
		lastAction = selectedActionCode; // remember action to compute reward
	}
	
	// return  0 : no signal
	// return  1 : if positive reward
	// return -1 : if negative reward
	override int retRewardFromLastAction() {
		if (lastAction == correctAction) {
			return 1;
		}
		return -1;
	}
	
	override void setGlobalIterationCounter(long iterationCnt) {
	}
}








// from arcish.d
class Map2d {
    public long[] arr;
    public long width;

    public final this(Vec2i size) {
        width = size.x;
        foreach (iy; 0..size.y) {
            foreach (ix; 0..size.x) {
                arr ~= 0;
            }
        }
    }

    public final Vec2i retSize() {
        long height = arr.length / width;
        return Vec2i(width, height);
    }

    public final long readAt(Vec2i pos) {
		if (pos.x < 0 || pos.x >= retSize().x || pos.y < 0 || pos.y >= retSize().y) {
			// TODO : return default value of datatype
			return 0;
		}
        return arr[ pos.x + width*pos.y ];
    }

    public final void writeAt(long v, Vec2i pos) {
		if (pos.x < 0 || pos.x >= retSize().x || pos.y < 0 || pos.y >= retSize().y) {
			return;
		}
        arr[ pos.x + width*pos.y ] = v;
    }
}

Map2d map_submap(Vec2i corner, Vec2i size, Map2d map) {
	Map2d mapRes = new Map2d(size);
	for (int iy=0;iy < size.y; iy++) {
		for (int ix=0;ix < size.x; ix++) {
			long v = map.readAt(Vec2i(corner.x+ix, corner.y+iy));
			mapRes.writeAt(v, Vec2i(ix, iy));
		}
	}
	return mapRes;
}

double map_calcSimilarity(Map2d a, Map2d b) {
	long cnt = 0;
	long cntPos = 0;
	for (int iy=0;iy < a.retSize().y; iy++) {
		for (int ix=0;ix < a.retSize().x; ix++) {
			if (a.readAt(Vec2i(ix, iy)) == b.readAt(Vec2i(ix, iy))) {
				cntPos+=1;
			}
			cnt+=1;
		}
	}
	return cast(double)cntPos / cast(double)cnt;
}

Map2d copy(Map2d arg) {
	Map2d imgRes = new Map2d(arg.retSize());
	
	for (int iy=0;iy < arg.retSize().y; iy++) {
		for (int ix=0;ix < arg.retSize().x; ix++) {
			long v = arg.readAt(Vec2i(ix, iy));
			imgRes.writeAt(v, Vec2i(ix, iy));
		}
	}
	
	return imgRes;
}

string map_convToStr(Map2d map) {
	string s = "";
	for(long iy=0;iy<map.retSize().y;iy++) {
		string sLine = "";
		for(long ix=0;ix<map.retSize().x;ix++) {
			sLine ~= format("%d", map.readAt(Vec2i(ix, iy)));
		}
		
		s ~= (sLine ~ "\n");
	}
	return s;
}




struct Vec2i {
    long x;
    long y;

    public final this(long x, long y) {
        this.x = x;
        this.y = y;
    }
}

Vec2i sub(Vec2i lhs, Vec2i rhs) {
	return Vec2i(lhs.x-rhs.x, lhs.y-rhs.y);
}





// serialize a Map2d as a one hot encoded array
Vec conv_map2d_to_arrOneHot(Map2d map, int nColors) {
	Vec res = new Vec([]);
	
	foreach (iy; 0..map.retSize().y) {
		foreach (ix; 0..map.retSize().x) {
			res = append(res, oneHotEncode(map.readAt(Vec2i(ix, iy)), nColors));
		}
	}
	
	return res;
}







// TODO  : use cortial thingy with actual ARC puzzle where a line has to get drawn

// simple ARC-like task: draw a line to the right side
//
//
//        ... draw  ... right ... draw  ...
//        .A.  ->   .X.  ->   X..  ->   XX.
//        ...       ...       ...       ...
//   
// reward            +1                 +1


// simple cursor environment for ARC solver
class SimpleCursor0Env : EnvAbstract {
	int windowExtend = 3; // CONFIG

	Map2d imgScratchpad = new Map2d(Vec2i(10, 10));
	
	// image of rightside of ARC puzzle to learn from
	// is null if it is in inference mode
	Map2d imgRightside = new Map2d(Vec2i(10, 10));
	
	Vec2i posCursorCenter = Vec2i(1, 1); // current center position of the cursor
	
	long iterationCnt = 0;
	
	bool wasLastActionChangeWrite = false; // (private)
	
	this() {
	}
	
	override Vec receivePerception() {
		
		// cut out the view of "imgScratchpad"
		Map2d imgSub = map_submap(sub(posCursorCenter, Vec2i((windowExtend-1)/2, (windowExtend-1)/2)), Vec2i(windowExtend, windowExtend), imgScratchpad);
		
		Vec perceivedStimulus = conv_map2d_to_arrOneHot(imgSub, 12);
		
		return perceivedStimulus; // return perceivedStimulus as output from the environment
	}
	
	
	override void doAction(string selectedActionCode) {
		wasLastActionChangeWrite = false;
		if (selectedActionCode == "^move(-1, 0)") {
			posCursorCenter.x -= 1;
		}
		else if (selectedActionCode == "^move(1, 0)") {
			posCursorCenter.x += 1;
		}
		else if (selectedActionCode == "^move(0, -1)") {
			posCursorCenter.y -= 1;
		}
		else if (selectedActionCode == "^move(0, 1)") {
			posCursorCenter.y += 1;
		}
		else if (selectedActionCode == "^draw(2)") { // draw color
			writeln(format("draw at pos=<%d %d>", posCursorCenter.x, posCursorCenter.y));
		
			long valBefore = imgScratchpad.readAt(posCursorCenter);
			
			// writeln( map_convToStr(imgScratchpad) );	 // DEBUG
						
			imgScratchpad.writeAt(2, posCursorCenter);
			
			// writeln( map_convToStr(imgScratchpad) );	 // DEBUG
			
			
			wasLastActionChangeWrite = true; //valBefore != 2; // we did change pixel color if values are different
		}
	}
	
	// return  0 : no signal
	// return  1 : if positive reward
	// return -1 : if negative reward
	override int retRewardFromLastAction() {
		// reward by comparing imgScratchpad to imgRightside when a color was draw
		if (wasLastActionChangeWrite) {
			if (imgScratchpad.readAt(posCursorCenter) == imgRightside.readAt(posCursorCenter)) {
				return 1;
			}
		}
		
		// else we return a empty reward
		return 0;
	}
	
	override void setGlobalIterationCounter(long iterationCnt) {
		this.iterationCnt = iterationCnt;
	}
}










// lab idea: use multiplication mask as attention

// idea: use multiplication mask as attention. this form of attention allows the units to mask out certain channels
double[] calcSimArrByAttention(Vec perceivedStimulus, CtxZZZ ctx) {
	double[] arrUnitSim = [];
	
	foreach (iUnit; ctx.units) {
		Vec postAttention;
		if (true) { // use attention?
			postAttention = mulComponents(perceivedStimulus, iUnit.attentionMask);
		}
		else {
			// COMMENTED BECAUSE NOT TRIED
		
			// else post attention is just perceivedStimulus
			postAttention = perceivedStimulus;
		}
		
		Vec key = iUnit.v; // key vector which we take from the pattern for which the unit is looking for
		
		double voteSimScalar = dot(key, postAttention);
		
		arrUnitSim ~= voteSimScalar;	
	}
		
	return arrUnitSim;
}


//	double[] function(Vec perceivedStimulus, CtxZZZ ctx)  calcSimFn;
//	calcSimFn = &calcSimArrByAttention;
//	
//	double[] arrSim = calcSimFn(null, null);

class SoftMaxSimilarityAttentionCalculationStrategy : SimilarityCalculationStrategy {
	override double[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx) {
		return calcSimArrByAttention(stimulus, ctx);
	}
}




// learning algorithm next step:
//    reward attention pattern of the winner unit and punish attention pattern of the looser units
//    TODO TODO TODO



// TODO LOW : attention: use attention and also implement attentionMask rewarding + punishment to allow the algorithm to update the attention masks
//    TODO : use attention in main loop
//    TODO : implement update of attention masks of winner unit and looser units of voting






// DONE : use "predictedReward" for action selection based on stimuli!




// TODO HIGH : implement use code of SimpleCursor0Env  for extremly simple drawing task

// for ARC-AGI
class ImagePair {
	Map2d imgLeftside;
	Map2d imgRightside;
	
	final this(Map2d imgLeftside, Map2d imgRightside) {
		this.imgLeftside = imgLeftside;
		this.imgRightside = imgRightside;
	}
}

// for ARC-AGI
class ImagePairsCtx {
	ImagePair[] imagePairs;
}

// lab : drawing task
void IDEA_LAB__drawingTaskSimpleA() {

	long globalIterationCounter = 0;
	
	CortialAlgoithm_LearnerCtx learner = new CortialAlgoithm_LearnerCtx();
	
	
	learner.env = new SimpleCursor0Env(); // we set the environment to simple cursor for ARC environment
	
	
	
	learner.column.availableActions = [];
	//learner.column.availableActions ~= "^move(-1, 0)";
	learner.column.availableActions ~= "^move(1, 0)";
	//learner.column.availableActions ~= "^move(0, -1)";
	//learner.column.availableActions ~= "^move(0, 1)";
	learner.column.availableActions ~= "^draw(2)";
	
	
	// code which encodes the task
	string[] taskCode = ["3"];
	
	
	ImagePairsCtx imagePairs = new ImagePairsCtx();
	
	// add actual pair for UNITTEST
	{
		Map2d imgLeftside;
		Map2d imgRightside;
		
		Vec2i sizeImg = Vec2i(5, 5);
		
		imgLeftside = new Map2d(sizeImg);
		imgLeftside.writeAt(1, Vec2i(1, 1));
		imgRightside = new Map2d(sizeImg);
		imgRightside.writeAt(2, Vec2i(1, 1));
		
		if (taskCode[0] == "1") {
			imgRightside.writeAt(2, Vec2i(2, 1));
		}
		
		if (taskCode[0] == "2") {
			imgRightside.writeAt(2, Vec2i(2, 1));
			imgRightside.writeAt(2, Vec2i(3, 1));
		}
		
		if (taskCode[0] == "3") {
			imgRightside.writeAt(2, Vec2i(2, 1));
			imgRightside.writeAt(2, Vec2i(3, 1));
			imgRightside.writeAt(2, Vec2i(4, 1));
		}
		
		ImagePair createdImagePair = new ImagePair(imgLeftside, imgRightside);
		imagePairs.imagePairs ~= createdImagePair;
	}
	
	
	
	// drawing task
	
	foreach (itImagePair; imagePairs.imagePairs) {
	
		writeln("");
		writeln("task: learn based on image pair ...");
		
		
		
		// process processLearnDrawA BEGIN: we let here the learner learn the actual task for the image pair	
		
		
	
		
		for (long itAttemptForPair=0; itAttemptForPair < 10000; itAttemptForPair++) {
			
			writeln(format("task:    itAttempt=%d", itAttemptForPair));
			
			learner.resetColumnStates();

			learner.env = new SimpleCursor0Env();
			
			(cast(SimpleCursor0Env)(learner.env)).imgRightside = itImagePair.imgRightside;
			
			// first we need to reset the scratchpad image to imgLeftside
			(cast(SimpleCursor0Env)(learner.env)).imgScratchpad = copy(itImagePair.imgLeftside);
			
			// we need to set the cursor position
			(cast(SimpleCursor0Env)(learner.env)).posCursorCenter = Vec2i(1, 1);
			
			
			
			
			// debug image to terminal
			{
				writeln("");
				writeln( map_convToStr((cast(SimpleCursor0Env)(learner.env)).imgScratchpad) );
				
			}
			
			
			
			
			for (long cntIterationOfTaskAttempt=0; cntIterationOfTaskAttempt<7; cntIterationOfTaskAttempt++) {
				
				writeln(format("task:       cntIterationOfTaskAttempt=%d", cntIterationOfTaskAttempt));
				
				// (learner iteration toegther with environment iteration)
				
				learner.learnerSyncronousAndEnviromentStep(globalIterationCounter);
				globalIterationCounter += 1;
			}
			
			// give leanrer a chance to learn from last observation
			learner.finish();
			
			
			// debug image to terminal
			{
				writeln("");
				writeln( map_convToStr((cast(SimpleCursor0Env)(learner.env)).imgScratchpad) );	
			}
			
			{
				writeln("");
				writeln( map_convToStr(itImagePair.imgRightside) );				
			}

			
			
			
			// now we check if the imgRightside got arch(ie)ved and how close we are to it
			double similarityOfImages = map_calcSimilarity((cast(SimpleCursor0Env)(learner.env)).imgScratchpad, itImagePair.imgRightside);
			
			writeln(format("task: attempt: similarityOfAttempt=%f", similarityOfImages));
				
			if (similarityOfImages >= 1.0-1e-6) { // is the result perfect?
				// this means that we did learn the task from the image-pair successfully
				
				writeln("task: 100% match!");
				
				//exit(0); // DEBUG
				
				break; // we break out of the loop to learn from this image pair
			}
		}
		
		// process processLearnDrawA END
	}
	
	
	// * now we check if the task is solvable with the learned model
	//   
	//   algorithm: we simply iterate over all pairs and see if the learner can successfully solve it in inference mode
	
	// TODO TODO TODO
	// TODO TODO TODO
	// TODO TODO TODO
	// TODO TODO TODO
	// TODO TODO TODO
	
	
	
	
	// debug output of the run of this task
	// TODO
	
	
	
}






