

// See https://aka.ms/new-console-template for more information
//Console.WriteLine("Hello, World!");

EntryA.entryA();


public static class EntryA {
	public static void entryA() {
		LabA.IDEA_LAB__drawingTaskSimpleA();
	}
}




// basic idea: 
// we have units
//    traits of units: program unit's with traits, such as need to survive by competing for resources (for example up-votes when a stimuli matches to the pattern detector its looking out for). 
//
// all is orchestrated by a "soup manager" - which determines the overall task of the units
public class UnitB {
	public Vec v; // vector to look out
	
	public string guid; // unique id which identifies this unit
	
	public UnitEvidence unitEvidence = new UnitEvidence();
	
	
	// attribute for stimuli + action mapping
	public string actionCode; // associated action code
	public Vec consequenceVec; // vector of the consequence
	public int predictedReward = 0; // predicted reward which is associated with the consequence after the action
	
	
	public Vec attentionMask; // attention mask for which this unit is looking out for
	
	
	public UnitB(string guid) {
		this.guid = guid;
	}
}

public static class UnitUtils {
	public static string retDebugStrOfUnit(UnitB unit) {
		return string.Format("v ={0}\nactionCode={1}\npredictedReward={2}\n\n", string.Join(",", unit.v.arr), unit.actionCode, unit.predictedReward);
	}
}

// evidence for the unit
public class UnitEvidence {
	public long positiveMatchCnt = 0; // matching counter for positive matches
	
	// TODO ::: Time lastPositiveMatchTime = null; // time of last positive matching	
	public void addPositive() {
		positiveMatchCnt += 1;
	}
}


// context which contains units
public class CtxZZZ {
	public List<UnitB> units = new List<UnitB>();
}

public class ZZZx {
	// context we are using
	public CtxZZZ ctx = new CtxZZZ();

	public void identifyAndLearn(Vec v) {
		double bestSim = -1.0;
		UnitB bestUnit = null;
		
		foreach (UnitB itUnit in ctx.units) {
			double sim = VecUtils.calcCosineSim(v, itUnit.v);
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
			
			
			ctx.units.Add(createdUnit);
			// TODO : care about AIKR here
		}
	}
}










// strategy to calculate similarity
public abstract class SimilarityCalculationStrategy {
	public abstract double[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx);
}

// soft computing similarity
public class SoftMaxSimilarityCalculationStrategy : SimilarityCalculationStrategy {
	public override double[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx) {
		double[] arrSim = new double[ctx.units.Count];
		
		int idx=0;
		foreach (UnitB itUnit in ctx.units) {
			double sim = VecUtils.calcCosineSim(stimulus, itUnit.v);
			double sim2 = (sim+1.0) * 0.5; // map to 0.0 1.0 range
			arrSim[idx] = sim2;
			idx++;
		}
		
		return arrSim;
	}
}

// hard similarity
public class HardMaxSimilarityCalculationStrategy : SimilarityCalculationStrategy {
	public override double[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx) {
		double bestSim = -1.0;
		long bestUnitIdx = -1;
	
		int itIdx = 0;
		foreach (UnitB itUnit in ctx.units) {
			double sim = VecUtils.calcCosineSim(stimulus, itUnit.v);
			if (sim > bestSim) {
				bestSim = sim;
				bestUnitIdx = itIdx;
			}
			itIdx++;
		}
		
		double[] arrSim = new double[ctx.units.Count];
		arrSim[bestUnitIdx] = (bestSim+1.0) * 0.5;
		
		return arrSim;
	}
}






// temporarily holds the result of the planning
public class AlgorithmResult__Planning {
	public string firstActionActionCode; // actionCode of the first action which the planning algorithm has computed
	public double expectedRewardSum = 0.0; // exptected reward sum for the selected action
}

// use of predicted input for 
// mechanism:
// a) feed input X to column to predict input X which follows, together with the action and reward
// b) goto a)

// TODO IDEA: we could predict the next input and the reward with NN which are trained.

public static class CortialCore {
	public static AlgorithmResult__Planning LAB__cortialAlgorithm__planning_A(Vec stimulus, ColumnCtxA columnCtx, int nPlanningDepth) {



		//int nPlanningDepth = 1; // how many iterations are done for planning
		//int nPlanningDepth = 4; // how many iterations are done for planning
	

		Vec iteratedStimulus = stimulus;
	
		double expectedRewardSum = 0.0; // sum of rewards of the "path"
		string firstActionActionCode = null;
	
		if (columnCtx.ctx.units.Count > 0) { // there must be units to vote on
			for(int itPlanningDepth=0;itPlanningDepth<nPlanningDepth;itPlanningDepth++) {
				// vote for best unit
				VotingWeightsOfUnits votingWeights = iteratedPlanning__voteUnitsAsVotingWeights(iteratedStimulus, columnCtx);
			
				// select winner unit weights
				VotingWeightsOfUnits votingWeightsAfterSelectingWinner = iteratedPlanning__selectWinnerUnitVector(votingWeights);
			
				if (itPlanningDepth == 0) {
					firstActionActionCode = columnCtx.ctx.units[ calcIndexWithHighestValue(votingWeightsAfterSelectingWinner) ].actionCode;
				}
			
				expectedRewardSum += (calcWeightedPredictedReward(votingWeightsAfterSelectingWinner.v, columnCtx) * Math.Exp(-(double)itPlanningDepth * 0.9));
			
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

	
	public static VotingWeightsOfUnits iteratedPlanning__voteUnitsAsVotingWeights(Vec stimulus, ColumnCtxA columnCtx) {
	
		SimilarityCalculationStrategy similarityCalcStrategy;
		similarityCalcStrategy = new SoftMaxSimilarityCalculationStrategy();
		//similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy(); // use attention strategy
		double[] arrSim = similarityCalcStrategy.calcMatchingScore__by__stimulus(stimulus, columnCtx.ctx);
	
		return new VotingWeightsOfUnits( VecUtils.normalize( new Vec(arrSim) ) );
	}


	public static VotingWeightsOfUnits iteratedPlanning__selectWinnerUnitVector(VotingWeightsOfUnits votingWeights) {
	
		int maxIdx = 0;
		double maxValue = -double.MaxValue;
		int itIdx = 0;
		foreach (double itVal in votingWeights.v.arr) {
			if (itVal > maxValue) {
				maxValue = itVal;
				maxIdx = itIdx;
			}
			itIdx++;
		}
	
		Vec vecOneHot = VecUtils.oneHotEncode(maxIdx, votingWeights.v.arr.Length);
	
		return new VotingWeightsOfUnits(vecOneHot);
	}

	public static double calcWeightedPredictedReward(Vec v, ColumnCtxA columnCtx) {
		double res = 0.0;
		for (int idx=0; idx<v.arr.Length; idx++) {
			res += ( v.arr[idx] * columnCtx.ctx.units[idx].predictedReward );
		}
		return res;
	}

	public static Vec computePredictedVector(VotingWeightsOfUnits votingWeights, ColumnCtxA columnCtx) {
		Vec res = VecUtils.vecMakeByLength(columnCtx.ctx.units[0].consequenceVec.arr.Length);
	
		for (int idxUnit=0; idxUnit<columnCtx.ctx.units.Count; idxUnit++) {
			res = VecUtils.add( VecUtils.scale(columnCtx.ctx.units[idxUnit].consequenceVec, votingWeights.v.arr[idxUnit]), res);
		}
	
		return res;
	}



	public static int calcIndexWithHighestValue(VotingWeightsOfUnits votingWeights) {
		return VecUtils.calcHighestValueIdx(votingWeights.v);
	}



}

// typed helper class to give a vector which is the normalized weight a type
public class VotingWeightsOfUnits {
	public Vec v;
	
	public VotingWeightsOfUnits(Vec v) {
		this.v = v;
	}
}















// TODO : implement soft mapper of
// observed state + action -> effect state
//
// with using SoftMaxSimilarityCalculationStrategy to compute the blend of the interpolation to vote on best next action

// TODO TODO TOOD TODO






// context which maps to a single crtial column
public class ColumnCtxA {
	public CtxZZZ ctx = new CtxZZZ();
	
	public List<string> availableActions = new List<string>();
	
	public Vec lastPerceivedStimulus = null;
	public string lastSelectedAction = null;
	
	//  0 is no reward signal
	//  1 is positive reward
	// -1 is negative reward
	public int lastRewardFromEnvironment = 0;
}

/*

learner is based on following ideas:

* cortial algorithms as core methodology of substrate

* computing similarity between neurons :  modern hopfield neural networds
* unsupervised learning of prediction  :  hierachical temporal memory (HTM) theory

* TODO : decision making by voting of multiple column: TODO : searhc paper from hawkins

*/

// context of the cortial algorithm - learning and inference and interaction with the environment
public class CortialAlgoithm_LearnerCtx {
	public ColumnCtxA column = new ColumnCtxA();
	
	
	// reward statistics
	public long cntRewardPos = 0;
	public long cntRewardNeg = 0;
	
	
	public double paramRandomActionChance = 0.1; // probability to act with a random action in each timestep
	
	public double paramGoodEnoughSimThreshold = 0.95;
	

	public RngA rng = new RngA();
	
	
	public EnvAbstract env = null; // must be set externally
	
	
	
	public void resetColumnStates() {
		
		column.lastPerceivedStimulus = null;
		column.lastSelectedAction = null;
	
	}
	
	
	// synchronous step between learner and environment
	public void learnerSyncronousAndEnviromentStep(long globalIterationCounter) {
		
		Console.WriteLine("");
		Console.WriteLine("");
		Console.WriteLine("");
		
		// DEBUG units
		{
			Console.WriteLine("units:");
			foreach (UnitB itUnit in column.ctx.units) {
				Console.WriteLine("");
				Console.WriteLine(UnitUtils.retDebugStrOfUnit(itUnit));
			}
			Console.WriteLine("");
		}
		
		
		env.setGlobalIterationCounter(globalIterationCounter);
		
		Vec perceivedStimulus;
		
		perceivedStimulus = env.receivePerception();
		
		Console.WriteLine(string.Format("perceived stimulus= {0}", perceivedStimulus.arr));
		
		
		
		// build contigency from past observation, past action and current stimulus and update in memory
		buildContingencyFromPastObservation(perceivedStimulus);
		
		
		
		// STAGE: compare stimulus to prototypes and compute for each unit: similarity of stimulus to compared patttern of unit
		
		// use calc__action__byVotingMax to decide on the next action based on the observed stimulus state . then update the associations based on the observed effect state
		
		
		
		SimilarityCalculationStrategy similarityCalcStrategy;
		similarityCalcStrategy = new SoftMaxSimilarityCalculationStrategy();
		//similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy(); // use attention strategy
		double[] arrSim = similarityCalcStrategy.calcMatchingScore__by__stimulus(perceivedStimulus, column.ctx);
		
		// DEBUG
		Console.WriteLine("");
		Console.WriteLine("similarities to perceivedStimulus:");
		Console.WriteLine(arrSim);
		
		
		
		
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
		AlgorithmResult__Planning resPlanning = CortialCore.LAB__cortialAlgorithm__planning_A(perceivedStimulus, column, nPlanningDepth);
		
		string maxVotingActionCode = resPlanning.firstActionActionCode; // copy selected action code into temporary variable
		
		
		
		
		
		// DEBUG
		Console.WriteLine(string.Format("selected max firstActionActionCode={0}    expected future rewardSum={1}", resPlanning.firstActionActionCode, resPlanning.expectedRewardSum));
		
		string selectedActionCode = null;
		
		if (!(maxVotingActionCode is null)) {
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
			int idxSel = (int)rng.nextInteger(column.availableActions.Count);
			selectedActionCode = column.availableActions[idxSel];
			Console.WriteLine("DBG  random action is selected");
		}
		
		
		// DEBUG
		Console.WriteLine(string.Format("selected actionCode={0}", selectedActionCode));
		
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
				Console.WriteLine("column: received - NEGATIVE reward!");
			}
			else if (column.lastRewardFromEnvironment > 0) {
				Console.WriteLine("column: received + POSITIVE reward!");
			}
			
			// TODO : use reward before next cycle to reward/punish units
		}
		
		
		// stage: prepare next cycle
		column.lastSelectedAction = selectedActionCode;
		column.lastPerceivedStimulus = perceivedStimulus;
		
		
		
		
		// statistics: we output here statistics about the reward thus far
		{
			Console.WriteLine("");
			double rationOfReward = (double)cntRewardPos / (double)(cntRewardPos + cntRewardNeg);
			Console.WriteLine(string.Format("global: ratio of reward={0}", rationOfReward));
		}
	}
	
	// give the learner a chance to learn from the last step
	public void finish() {
		Vec perceivedStimulus;
		
		perceivedStimulus = env.receivePerception();
		
		Console.WriteLine(string.Format("perceived stimulus= {0}", VecUtils.convToStr(perceivedStimulus)));
		
		
		
		// build contigency from past observation, past action and current stimulus and update in memory
		buildContingencyFromPastObservation(perceivedStimulus);
	}
	
	
	
	
	
	
	// (private)
	//
	// build contigency from past observation, past action and current stimulus and update in memory
	//
	public void buildContingencyFromPastObservation(Vec perceivedStimulus) {
		// NOTE : we only update if it is similar enough to existing condition
		
		if (!(column.lastPerceivedStimulus is null)) {
			
			bool wasMatchFound = false;
			
			// search for match
			SimilarityCalculationStrategy similarityCalcStrategy = new SoftMaxSimilarityCalculationStrategy();
			double[] arrSim = similarityCalcStrategy.calcMatchingScore__by__stimulus(column.lastPerceivedStimulus, column.ctx);
			
			// DEBUG
			Console.WriteLine("");
			Console.WriteLine("sim to perceived contigency:");
			Console.WriteLine(VecUtils.convToStr(new Vec(arrSim)));
			
			
			
			// decide if the match is good enough
			{
				int idxBest = -1;
				double valBest = -2.0;
				for (int itIdx=0; itIdx<arrSim.Length; itIdx++) {
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
				UnitB createdUnit = new UnitB(Guid.NewGuid().ToString());
				createdUnit.v = column.lastPerceivedStimulus;
				createdUnit.attentionMask = VecUtils.vecMake(1.0, createdUnit.v.arr.Length); // attention mask which doesn't change any channels by default (to make testing easier)
				createdUnit.actionCode = column.lastSelectedAction;
				createdUnit.consequenceVec = perceivedStimulus;
				
				createdUnit.predictedReward = column.lastRewardFromEnvironment; // attach reward to be able to predict reward
				// TODO LOW : revise predictedReward via some clever formula when a good enough unit was found
				
				// reward created unit
				createdUnit.unitEvidence.addPositive();
				
				
				column.ctx.units.Add(createdUnit);
				// TODO : care about AIKR here
				
				Console.WriteLine("");
				Console.WriteLine("DBG :  created new unit by contingency");
			}
		}
	}
}



public static class ManualtestsA {
	public static void manualtest__softComputingAssocA() {

	
		CortialAlgoithm_LearnerCtx learner = new CortialAlgoithm_LearnerCtx();
	
	
		//learner.env	= new PerceptualDummy0Env(); // create dummy environment
		learner.env = new Simple0Env(); // create simple environment for unittesting
	
		learner.env = new SimpleAssoc0Env(); // simple association environment for 
	
	
	
		learner.column.availableActions = new List<string>();
		learner.column.availableActions.Add("^a");
		learner.column.availableActions.Add("^b");
		learner.column.availableActions.Add("^c");
	
	
		learner.resetColumnStates();
	
	
		for (long globalIterationCounter=0;globalIterationCounter<2000;globalIterationCounter++) {
			learner.learnerSyncronousAndEnviromentStep(globalIterationCounter);
			globalIterationCounter += 1;		
		}
	}	

}







public interface EnvAbstract {
	// perceive the perception from the environment
	Vec receivePerception();

	// do action in the environment
	void doAction(string selectedActionCode);
	
	// return  0 : no signal
	// return  1 : if positive reward
	// return -1 : if negative reward
	int retRewardFromLastAction();
	
	
	
	// helper to inform environment about current global iteration counter
	void setGlobalIterationCounter(long iterationCnt);
}

public class PerceptualDummy0Env : EnvAbstract {
	public long iterationCnt = 0; // (private) iteration counter : used to generate different dummy stimulus from the environment

	public PerceptualDummy0Env() {
	}
	
	public Vec receivePerception() {
		Map2d map0 = new Map2d(new Vec2i(3, 3));
		map0.writeAt(1, new Vec2i(iterationCnt % map0.retSize().x, 2));
		
		Vec perceivedStimulus = Map2dUtils.conv_map2d_to_arrOneHot(map0, 12);
		return perceivedStimulus;
	}
	
	public void doAction(string selectedActionCode) {		
		// do nothing
	}
	
	public int retRewardFromLastAction() {
		return 0;
	}
	
	public void setGlobalIterationCounter(long iterationCnt) {
		this.iterationCnt = iterationCnt;
	}
}


// very simple environment which rewards only one action and punishes all others
//
// is used to UNITTEST the core learner that it learns to pick the correct action
class Simple0Env : EnvAbstract {
	private string lastAction = null;
	
	public string correctAction = "^b";

	public Simple0Env() {
	}
	
	public Vec receivePerception() {
		return new Vec(new double[]{1.0}); // return dummy perception
	}
	
	public void doAction(string selectedActionCode) {		
		lastAction = selectedActionCode; // remember action to compute reward
	}
	
	// return  0 : no signal
	// return  1 : if positive reward
	// return -1 : if negative reward
	public int retRewardFromLastAction() {
		if (lastAction == correctAction) {
			return 1;
		}
		return -1;
	}
	
	public void setGlobalIterationCounter(long iterationCnt) {
	}
}


// simple environment which tests if a association of a color to a action is usccessfully learned by the learner
public class SimpleAssoc0Env : EnvAbstract {
	
	public string correctAction = null; // private
	
	public RngA rngEnv = new RngA();
	
	private string lastAction = null;
	
	public SimpleAssoc0Env() {
	}
	
	
	public Vec receivePerception() {
		Vec vecOut = VecUtils.vecMake(0.001, 3);
		
		List<string> actions = new List<string>(new string[]{"^a", "^b", "^c" });
		
		int selIdx = (int)rngEnv.nextInteger(actions.Count);
		vecOut.arr[selIdx] = 1.0; // encode perception so learner can associate stimuli with correct action
		correctAction = actions[selIdx]; // remember correct action
		
		return vecOut;
	}
	
	public void doAction(string selectedActionCode) {		
		lastAction = selectedActionCode; // remember action to compute reward
	}
	
	// return  0 : no signal
	// return  1 : if positive reward
	// return -1 : if negative reward
	public int retRewardFromLastAction() {
		if (lastAction == correctAction) {
			return 1;
		}
		return -1;
	}
	
	public void setGlobalIterationCounter(long iterationCnt) {
	}
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
public class SimpleCursor0Env : EnvAbstract {
	public int windowExtend = 3; // CONFIG

	public Map2d imgScratchpad = new Map2d(new Vec2i(10, 10));
	
	// image of rightside of ARC puzzle to learn from
	// is null if it is in inference mode
	public Map2d imgRightside = new Map2d(new Vec2i(10, 10));
	
	public Vec2i posCursorCenter = new Vec2i(1, 1); // current center position of the cursor
	
	public long iterationCnt = 0;
	
	public bool wasLastActionChangeWrite = false; // (private)
	
	public SimpleCursor0Env() {
	}
	
	public Vec receivePerception() {
		
		// cut out the view of "imgScratchpad"
		Map2d imgSub = Map2dUtils.map_submap(Vec2iUtils.sub(posCursorCenter, new Vec2i((windowExtend-1)/2, (windowExtend-1)/2)), new Vec2i(windowExtend, windowExtend), imgScratchpad);
		
		Vec perceivedStimulus = Map2dUtils.conv_map2d_to_arrOneHot(imgSub, 12);
		
		return perceivedStimulus; // return perceivedStimulus as output from the environment
	}
	
	
	public void doAction(string selectedActionCode) {
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
			Console.WriteLine(string.Format("draw at pos=<{0} {1}>", posCursorCenter.x, posCursorCenter.y));
			
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
	public int retRewardFromLastAction() {
		// reward by comparing imgScratchpad to imgRightside when a color was draw
		if (wasLastActionChangeWrite) {
			if (imgScratchpad.readAt(posCursorCenter) == imgRightside.readAt(posCursorCenter)) {
				return 1;
			}
		}
		
		// else we return a empty reward
		return 0;
	}
	
	public void setGlobalIterationCounter(long iterationCnt) {
		this.iterationCnt = iterationCnt;
	}
}










// lab idea: use multiplication mask as attention

// idea: use multiplication mask as attention. this form of attention allows the units to mask out certain channels



//	double[] function(Vec perceivedStimulus, CtxZZZ ctx)  calcSimFn;
//	calcSimFn = &calcSimArrByAttention;
//	
//	double[] arrSim = calcSimFn(null, null);

public class SoftMaxSimilarityAttentionCalculationStrategy : SimilarityCalculationStrategy {
	public override double[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx) {
		return calcSimArrByAttention(stimulus, ctx);
	}

	private static double[] calcSimArrByAttention(Vec perceivedStimulus, CtxZZZ ctx) {
		double[] arrUnitSim = new double[ctx.units.Count];
	
		int idx=0;
		foreach (UnitB iUnit in ctx.units) {
			Vec postAttention;
			if (true) { // use attention?
				postAttention = VecUtils.mulComponents(perceivedStimulus, iUnit.attentionMask);
			}
			else {
				// COMMENTED BECAUSE NOT TRIED
		
				// else post attention is just perceivedStimulus
				postAttention = perceivedStimulus;
			}
		
			Vec key = iUnit.v; // key vector which we take from the pattern for which the unit is looking for
		
			double voteSimScalar = VecUtils.calcDot(key, postAttention);
		
			arrUnitSim[idx] = voteSimScalar;
			idx++;
		}
		
		return arrUnitSim;
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
public class ImagePair {
	public Map2d imgLeftside;
	public Map2d imgRightside;
	
	public ImagePair(Map2d imgLeftside, Map2d imgRightside) {
		this.imgLeftside = imgLeftside;
		this.imgRightside = imgRightside;
	}
}

// for ARC-AGI
public class ImagePairsCtx {
	public List<ImagePair> imagePairs = new List<ImagePair>();
}

// lab : drawing task
public static class LabA {
	public static void IDEA_LAB__drawingTaskSimpleA() {

		long globalIterationCounter = 0;
	
		CortialAlgoithm_LearnerCtx learner = new CortialAlgoithm_LearnerCtx();
	
	
		learner.env = new SimpleCursor0Env(); // we set the environment to simple cursor for ARC environment
	
	
	
		learner.column.availableActions = new List<string>();
		//learner.column.availableActions ~= "^move(-1, 0)";
		learner.column.availableActions.Add("^move(1, 0)");
		//learner.column.availableActions ~= "^move(0, -1)";
		//learner.column.availableActions ~= "^move(0, 1)";
		learner.column.availableActions.Add("^draw(2)");
	
	
		// code which encodes the task
		string[] taskCode = new string[]{"2"};
	
	
		ImagePairsCtx imagePairs = new ImagePairsCtx();
	
		// add actual pair for UNITTEST
		{
			Map2d imgLeftside;
			Map2d imgRightside;
		
			Vec2i sizeImg = new Vec2i(5, 5);
		
			imgLeftside = new Map2d(sizeImg);
			imgLeftside.writeAt(1, new Vec2i(1, 1));
			imgRightside = new Map2d(sizeImg);
			imgRightside.writeAt(2, new Vec2i(1, 1));
		
			if (taskCode[0] == "1") {
				imgRightside.writeAt(2, new Vec2i(2, 1));
			}
		
			if (taskCode[0] == "2") {
				imgRightside.writeAt(2, new Vec2i(2, 1));
				imgRightside.writeAt(2, new Vec2i(3, 1));
			}
		
			if (taskCode[0] == "3") {
				imgRightside.writeAt(2, new Vec2i(2, 1));
				imgRightside.writeAt(2, new Vec2i(3, 1));
				imgRightside.writeAt(2, new Vec2i(4, 1));
			}
		
			ImagePair createdImagePair = new ImagePair(imgLeftside, imgRightside);
			imagePairs.imagePairs.Add(createdImagePair);
		}
	
	
	
		// drawing task
	
		foreach (ImagePair itImagePair in imagePairs.imagePairs) {
	
			Console.WriteLine("");
			Console.WriteLine("task: learn based on image pair ...");
		
		
		
			// process processLearnDrawA BEGIN: we let here the learner learn the actual task for the image pair	
		
		
	
		
			for (long itAttemptForPair=0; itAttemptForPair < 5000; itAttemptForPair++) {
			
				Console.WriteLine(string.Format("task:    itAttempt={0}", itAttemptForPair));
			
				learner.resetColumnStates();

				learner.env = new SimpleCursor0Env();
			
				((SimpleCursor0Env)(learner.env)).imgRightside = itImagePair.imgRightside;
			
				// first we need to reset the scratchpad image to imgLeftside
				((SimpleCursor0Env)(learner.env)).imgScratchpad = Map2dUtils.copy(itImagePair.imgLeftside);
			
				// we need to set the cursor position
				((SimpleCursor0Env)(learner.env)).posCursorCenter = new Vec2i(1, 1);
			
			
			
			
				// debug image to terminal
				{
					Console.WriteLine("");
					Console.WriteLine( Map2dUtils.map_convToStr(((SimpleCursor0Env)learner.env).imgScratchpad) );
				
				}
			
			
			
			
				for (long cntIterationOfTaskAttempt=0; cntIterationOfTaskAttempt<7; cntIterationOfTaskAttempt++) {
				
					Console.WriteLine(string.Format("task:       cntIterationOfTaskAttempt={0}", cntIterationOfTaskAttempt));
				
					// (learner iteration toegther with environment iteration)
				
					learner.learnerSyncronousAndEnviromentStep(globalIterationCounter);
					globalIterationCounter += 1;
				}
				
				// give leanrer a chance to learn from last observation
				learner.finish();
			
			
				// debug image to terminal
				{
					Console.WriteLine("");
					Console.WriteLine( Map2dUtils.map_convToStr(((SimpleCursor0Env)learner.env).imgScratchpad) );	
				}
			
				{
					Console.WriteLine("");
					Console.WriteLine( Map2dUtils.map_convToStr(itImagePair.imgRightside) );				
				}

			
			
			
				// now we check if the imgRightside got arch(ie)ved and how close we are to it
				double similarityOfImages = Map2dUtils.map_calcSimilarity(((SimpleCursor0Env)learner.env).imgScratchpad, itImagePair.imgRightside);
				
				Console.WriteLine(string.Format("task: attempt: similarityOfAttempt={0}", similarityOfImages));
				
				if (similarityOfImages >= 1.0-1e-6) { // is the result perfect?
					// this means that we did learn the task from the image-pair successfully
				
					Console.WriteLine("task: 100% match!");
				
					//exit(0); // DEBUG
				
					break; // we break out of the loop to learn from this image pair
				}

				int breakpointDEBUGhere5 = 1;
			}
		
			// process processLearnDrawA END
		}
	
	
		// * now we check if the task is solvable with the learned model
		//   
		//   algorithm: we simply iterate over all pairs and see if the learner can successfully solve it in inference mode
	
		// TODO TODO TODO
		// TODO TODO TODO
	
	
	
	
		// debug output of the run of this task
		// TODO
	
	
	
	}
}





















// Jeff Hawkin : "reference frames"  : explained in the book "The 1000 brains theory"









public struct Vec2i {
    public long x;
    public long y;

    public Vec2i(long x, long y) {
        this.x = x;
        this.y = y;
    }
}


public class Vec2iUtils {
    public static Vec2i sub(Vec2i lhs, Vec2i rhs) {
	    return new Vec2i(lhs.x-rhs.x, lhs.y-rhs.y);
    }
}








// from arcish.d
public class Map2d {
    public List<long> arr = new List<long>();
    public int width;

    public Map2d(Vec2i size) {
        width = (int)size.x;
        for(int iy = 0; iy < size.y; iy++) {
            for(int ix=0; ix < size.x; ix++) {
                arr.Add(0);
            }
        }
    }

    public Vec2i retSize() {
        long height = arr.Count / width;
        return new Vec2i(width, height);
    }

    public long readAt(Vec2i pos) {
		if (pos.x < 0 || pos.x >= retSize().x || pos.y < 0 || pos.y >= retSize().y) {
			// TODO : return default value of datatype
			return 0;
		}
        return arr[ (int)(pos.x + width*pos.y) ];
    }

    public void writeAt(long v, Vec2i pos) {
		if (pos.x < 0 || pos.x >= retSize().x || pos.y < 0 || pos.y >= retSize().y) {
			return;
		}
        arr[ (int)(pos.x + width*pos.y) ] = v;
    }
}

public static class Map2dUtils {
    public static Map2d map_submap(Vec2i corner, Vec2i size, Map2d map) {
	    Map2d mapRes = new Map2d(size);
	    for (int iy=0;iy < size.y; iy++) {
		    for (int ix=0;ix < size.x; ix++) {
			    long v = map.readAt(new Vec2i(corner.x+ix, corner.y+iy));
			    mapRes.writeAt(v, new Vec2i(ix, iy));
		    }
	    }
	    return mapRes;
    }

	
	public static Map2d copy(Map2d arg) {
		Map2d imgRes = new Map2d(arg.retSize());
	
		for (int iy=0;iy < arg.retSize().y; iy++) {
			for (int ix=0;ix < arg.retSize().x; ix++) {
				long v = arg.readAt(new Vec2i(ix, iy));
				imgRes.writeAt(v, new Vec2i(ix, iy));
			}
		}
		
		return imgRes;
	}

	public static string map_convToStr(Map2d arg) {
		string s = "";
		for(long iy=0;iy<arg.retSize().y;iy++) {
			string sLine = "";
			for(long ix=0;ix<arg.retSize().x;ix++) {
				sLine += string.Format("{0}", arg.readAt(new Vec2i(ix, iy)));
			}
		
			s += (sLine + "\n");
		}
		return s;
	}

	public static double map_calcSimilarity(Map2d a, Map2d b) {
		long cnt = 0;
		long cntPos = 0;
		for (int iy=0;iy < a.retSize().y; iy++) {
			for (int ix=0;ix < a.retSize().x; ix++) {
				if (a.readAt(new Vec2i(ix, iy)) == b.readAt(new Vec2i(ix, iy))) {
					cntPos+=1;
				}
				cnt+=1;
			}
		}
		return (double)cntPos / (double)cnt;
	}

	// serialize a Map2d as a one hot encoded array
	public static Vec conv_map2d_to_arrOneHot(Map2d arg, int nColors) {
		Vec res = new Vec(new double[0]);
		
		for(int iy=0;iy<arg.retSize().y;iy++) {
			for(int ix=0;ix<arg.retSize().x;ix++) {
				res = VecUtils.append(res, VecUtils.oneHotEncode((int)arg.readAt(new Vec2i(ix, iy)), nColors));
			}
		}
	
		return res;
	}
}






public class Vec {
	public double[] arr;

	public Vec(double[] arr) {
		this.arr = arr;
	}
}

public static class VecUtils {
	public static Vec append(Vec a, Vec b) {
		double[] arr = new double[a.arr.Length + b.arr.Length];
		for(int idx=0;idx<a.arr.Length;idx++) {
			arr[idx] = a.arr[idx];
		}
		for(int idx=0;idx<b.arr.Length;idx++) {
			arr[a.arr.Length + idx] = b.arr[idx];
		}
		return new Vec(arr);
	}

	public static Vec vecMake(double v, int size) {
		double[] arr = new double[size];
		for (int idx=0;idx<size;idx++) {
			arr[idx] = v;
		}
		return new Vec(arr);
	}

	public static Vec vecMakeByLength(int size) {
		return new Vec(new double[size]);
	}
	
	public static Vec add(Vec a, Vec b) {
		double[] arr = new double[a.arr.Length];
		for (long i=0; i<a.arr.Length; i++) {
			arr[i] = a.arr[i] + b.arr[i];
		}
		return new Vec(arr);
	}

	public static Vec scale(Vec v, double s) {
		double[] arr = new double[v.arr.Length];
		for(int idx=0;idx<v.arr.Length;idx++) {
			arr[idx] = v.arr[idx] * s;
		}
		return new Vec(arr);
	}

	public static Vec mulComponents(Vec a, Vec b) {
		double[] arrRes = new double[a.arr.Length];
		for(long idx=0;idx<a.arr.Length;idx++) {
			arrRes[idx] = a.arr[idx]*b.arr[idx];
		}
		return new Vec(arrRes);
	}

	public static double calcCosineSim(Vec a, Vec b) {
		return calcDot(a, b) / (calcL2Norm(a)*calcL2Norm(b));
	}

	public static double calcDot(Vec a, Vec b) {
		double v=0.0;
		for(int idx=0;idx<a.arr.Length;idx++) {
			v += (a.arr[idx]*b.arr[idx]);
		}
		return v;
	}

	
	public static Vec normalize(Vec v) {
		double l = calcL2Norm(v);
		return VecUtils.scale(v, 1.0/l);
	}


	public static double calcL2Norm(Vec arg) {
		return Math.Sqrt(calcDot(arg, arg));
	}

	public static int calcHighestValueIdx(Vec arg) {
		int highestIdx = 0;
		double highestVal = arg.arr[0];
		for(int idx=0;idx<arg.arr.Length;idx++) {
			if (arg.arr[idx] > highestVal) {
				highestVal = arg.arr[idx];
				highestIdx = idx;
			}
		}
		return highestIdx;
	}

	public static Vec oneHotEncode(int symbol, int size) {
		double[] arr = new double[size];
		arr[symbol] = 1.0;
		return new Vec(arr);
	}

	public static string convToStr(Vec arg) {
		string[] arrStr = new string[arg.arr.Length];
		for(int idx=0;idx<arg.arr.Length;idx++) {
			arrStr[idx] = string.Format("{0}", arg.arr[idx]);
		}
		return string.Join(", ", arrStr);
	}
}







public class RngA {
    public double v = 0.01;

    // returns in range 0.0;1.0
    public double nextReal() {
        v += 0.01;
        return (1.0 + Math.Cos(v*10000000000.0)) * 0.5;
    }
	
	public long nextInteger(long max) {
		return (long)(nextReal() * max);
	}
}

public static class RngUtils {
	public static double[] genRngVec(long size, RngA rng) {
		double[] res = new double[size];
		for(long it=0; it<size; it++) {
			res[it] = rng.nextReal()*2.0 - 1.0;
		}
		return res;
	}
}




