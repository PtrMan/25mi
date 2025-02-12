

// pattern together with attention mask and usage counters
public class UnitPattern {
	public UnitEvidence patternEvidence = new UnitEvidence();
	
	public Vec attentionMask; // attention mask for which this unit is looking out for

	public string guid; // unique id which identifies this pattern

	public UnitPattern(string guid) {
		this.guid = guid;
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
	
	//public UnitEvidence unitEvidence = new UnitEvidence();
	
	
	// attribute for stimuli + action mapping
	public string actionCode; // associated action code
	public Vec consequenceVec; // vector of the consequence
	public int predictedReward = 0; // predicted reward which is associated with the consequence after the action
	
	
	//public Vec attentionMask; // attention mask for which this unit is looking out for
	
	
	public UnitB(string guid) {
		this.guid = guid;
	}

	public List<UnitPattern> patterns = new List<UnitPattern>();
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

public static class LearnerUtils {
	public static Vec extractSim(UnitPatternSim[] arrUnitPatternSims) {
		// extract only similarity values
		double[] arrSim = new double[arrUnitPatternSims.Length];
		{
			for(int idx=0; idx<arrUnitPatternSims.Length; idx++) {
				arrSim[idx] = arrUnitPatternSims[idx].sim;
			}
		}
		return  new Vec(arrSim);
	}
}

// context which contains units
public class CtxZZZ {
	public List<UnitB> units = new List<UnitB>();
}

public class ZZZx {
	// context we are using
	public CtxZZZ ctx = new CtxZZZ();
}










// strategy to calculate similarity
public abstract class SimilarityCalculationStrategy {
	public abstract UnitPatternSim[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx);
}



/* commented because not up to date and not used
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
*/





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
	// /param nPlanningDepth  how many iterations are done for planning
	public static AlgorithmResult__Planning LAB__cortialAlgorithm__planning_A(Vec stimulus, CortialAlgoithm_LearnerCtx learnerCtx, int nPlanningDepth) {
		
		Vec iteratedStimulus = stimulus;
		
		double expectedRewardSum = 0.0; // sum of rewards of the "path"
		string selFirstActionCode = null;

		
		
		for(int itPlanningDepth=0;itPlanningDepth<nPlanningDepth;itPlanningDepth++) {


			// use the predictive NN to find the action which leads to the highest future reward, given the "iteratedStimulus"
			string selActionCodeFromPredictiveNn = null;
			double highestFutureRewardFromPredictiveNn = -double.MaxValue;
			{
				if (learnerCtx.predictiveNn.ctx.units.Count > 0) { // we can only do this if there are units to vote on

					foreach (string itActionCode in  learnerCtx.columns[0].availableActions) {
						Tuple<Vec, double> resPredictiveNn = learnerCtx.predictiveNn.calcNextObservationStateAndRewardGivenObservationAndAction(iteratedStimulus, itActionCode);
						if (resPredictiveNn.Item2 > highestFutureRewardFromPredictiveNn) {
							highestFutureRewardFromPredictiveNn = resPredictiveNn.Item2;

							// for DEBUGGING
							if (resPredictiveNn.Item2 > 0) {
								int breakpointDEBUG0 = 5;
							}

							selActionCodeFromPredictiveNn = itActionCode;
						}
					}

				}
			}

			
			// used to collect the votes for the actionCode from all columns, given "iteratedStimulus"
			Dictionary<string, double> actionCodeVotes = new Dictionary<string, double>();
			// initialize to no votes
			foreach (string itActionCode in learnerCtx.columns[0].availableActions) {
				actionCodeVotes[itActionCode] = 0.0;
			}

			Dictionary<string, double> expectedRewardFromColumnVotesByAction = new Dictionary<string, double>();
			// initialize to no votes
			foreach (string itActionCode in learnerCtx.columns[0].availableActions) {
				expectedRewardFromColumnVotesByAction[itActionCode] = 0;
			}

			//double expectedRewardFromColumnVotes = 0.0;

			foreach (ColumnCtxA itColumn in learnerCtx.columns) {

				// TODO : dataflow for prediction may need global  mapping mechanism to go from predictedOutput to global input which is used as virtual stimulus for feedback

				if (itColumn.ctx.units.Count > 0) { // there must be units to vote on	
			
					// vote for best unit
					VotingWeightsOfUnits votingWeights = iteratedPlanning__voteUnitsAsVotingWeights(iteratedStimulus, itColumn);
			
					// select winner unit weights
					VotingWeightsOfUnits votingWeightsAfterSelectingWinner = iteratedPlanning__selectWinnerUnitVector(votingWeights);
			
					//if (itPlanningDepth == 0)
					{
						string selActionCodeFromColumn = itColumn.ctx.units[ calcIndexWithHighestValue(votingWeightsAfterSelectingWinner) ].actionCode;
						actionCodeVotes[selActionCodeFromColumn] = actionCodeVotes[selActionCodeFromColumn] + 1; // upvote the action

						//firstActionActionCode = itColumn.ctx.units[ calcIndexWithHighestValue(votingWeightsAfterSelectingWinner) ].actionCode;

						expectedRewardFromColumnVotesByAction[selActionCodeFromColumn] += calcWeightedPredictedReward(votingWeightsAfterSelectingWinner.v, itColumn.ctx);
					}
					
					//expectedRewardFromColumnVotes += (calcWeightedPredictedReward(votingWeightsAfterSelectingWinner.v, itColumn.ctx) * Math.Exp(-(double)itPlanningDepth * 0.9));
					
					// compute prediction of predicted output by vector
					//Vec vecPredicted = computePredictedVector(votingWeightsAfterSelectingWinner, itColumn.ctx);
			
					//iteratedStimulus = vecPredicted; // feed as stimulus for next iteration
				}
			}

			// vote on action decided by all columns
			string columnVoteActionCode = null;
			{				
				double votesMax = -double.MaxValue;
				foreach (KeyValuePair<string, double> itKeyValue in actionCodeVotes) {
					if (itKeyValue.Value > votesMax) {
						columnVoteActionCode = itKeyValue.Key;
						votesMax = itKeyValue.Value;
					}
				}
			}


			// decide on if we want to select the action by maximizing the future reward by using the predictive NN or if we want to decide based on votes from the columns
			//
			// here we randomly sample the action
			string selActionCode = null;
			double configSelStrategyProb = 0.70; // select by column or by predictive NN
			// random variable 0.5 doesnt work :( yet
			if (learnerCtx.rng.nextReal() < configSelStrategyProb) {
				selActionCode = columnVoteActionCode;

				expectedRewardSum += expectedRewardFromColumnVotesByAction[columnVoteActionCode] * Math.Exp(-(double)itPlanningDepth * 0.9); // better
			}
			else {
				selActionCode = selActionCodeFromPredictiveNn;
				expectedRewardSum += highestFutureRewardFromPredictiveNn * Math.Exp(-(double)itPlanningDepth * 0.9);
			}

			// we only care about selecting the first action in the sequence for decision making
			{
				if (itPlanningDepth == 0) {
					selFirstActionCode = selActionCode;
				}
			}
			
			// now we need to predict the actual next state based on the state and the chosen action
			{
				Tuple<Vec, double> resPredictiveNn = learnerCtx.predictiveNn.calcNextObservationStateAndRewardGivenObservationAndAction(iteratedStimulus, selActionCode);
				if (resPredictiveNn != null) {
					iteratedStimulus = resPredictiveNn.Item1;
				}
			}
		}
		
		
		// now we have a action "firstActionActionCode" which leads to a possible path with expected reward = "expectedRewardSum"
	
		// we have to do this a few times and select the action which gives us the highest expected reward
		// this is implemented in a outer loop

		AlgorithmResult__Planning res = new AlgorithmResult__Planning();
		res.firstActionActionCode = selFirstActionCode;
		res.expectedRewardSum = expectedRewardSum;
	
		return res;
	}

	
	public static VotingWeightsOfUnits iteratedPlanning__voteUnitsAsVotingWeights(Vec stimulus, ColumnCtxA columnCtx) {
	
		SimilarityCalculationStrategy similarityCalcStrategy;
		similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy();
		//similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy(); // use attention strategy
		UnitPatternSim[] arrUnitPatternSims = similarityCalcStrategy.calcMatchingScore__by__stimulus(stimulus, columnCtx.ctx);

		Vec vecSim = LearnerUtils.extractSim(arrUnitPatternSims); // extract only similarity values

		return new VotingWeightsOfUnits( VecUtils.normalize(vecSim) );
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

	public static double calcWeightedPredictedReward(Vec v, CtxZZZ ctx) {
		double res = 0.0;
		for (int idx=0; idx<v.arr.Length; idx++) {
			res += ( v.arr[idx] * ctx.units[idx].predictedReward );
		}
		return res;
	}

	public static Vec computePredictedVector(VotingWeightsOfUnits votingWeights, CtxZZZ ctx) {
		Vec res = VecUtils.vecMakeByLength(ctx.units[0].consequenceVec.arr.Length);
	
		for (int idxUnit=0; idxUnit<ctx.units.Count; idxUnit++) {
			res = VecUtils.add( VecUtils.scale(ctx.units[idxUnit].consequenceVec, votingWeights.v.arr[idxUnit]), res);
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



// helpers for common functionality to create units, etc.
public static class UnitPatternUtils {
	// /param v vector to detect by the unit
	public static UnitB makeUnitByStimulus(Vec v, string actionCode, Vec consequenceVec, int predictedReward) {
		
        // we add a new unit
		UnitB createdUnit = new UnitB(Guid.NewGuid().ToString());
		createdUnit.v = v;

		UnitPattern createdUnitPattern = new UnitPattern(Guid.NewGuid().ToString());
		createdUnitPattern.attentionMask = VecUtils.vecMake(1.0, createdUnit.v.arr.Length); // attention mask which doesn't change any channels by default (to make testing easier)                
		createdUnit.patterns.Add(createdUnitPattern);
				
		createdUnit.actionCode = actionCode;
		createdUnit.consequenceVec = consequenceVec;
				
		createdUnit.predictedReward = predictedReward; // attach reward to be able to predict reward
                                                                        // TODO LOW : revise predictedReward via some clever formula when a good enough unit was found

        // reward created pattern
		createdUnitPattern.patternEvidence.addPositive();

		return createdUnit;
	}
}




// idea: a NN which predicts the next complete environmental observation-state and reward from a given state and a action
//
//       will be used for "planning", by chaining together the observation state by the best action
public class PredictiveNn {
	// context used to store the units
	public CtxZZZ ctx = new CtxZZZ();

	public double paramGoodEnoughSimThreshold = 0.95;

	// learn (lastPerceivedStimulus, action) =/> <consequenceVec, predictedReward>
	public void learn(Vec lastPerceivedStimulus, Vec consequenceVec, string actionCode, int predictedReward) {
		
		if (!(lastPerceivedStimulus is null)) {
			
			bool wasMatchFound = false;
			
			// search for match
			SimilarityCalculationStrategy similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy();
            UnitPatternSim[] arrUnitPatternSims = similarityCalcStrategy.calcMatchingScore__by__stimulus(lastPerceivedStimulus, ctx);

            // DEBUG
            //Console.WriteLine("");
			//Console.WriteLine("sim to perceived contigency:");
			//Console.WriteLine(VecUtils.convToStr(new Vec(arrSim)));
			
			
			
			// decide if the match is good enough
			{
				int idxBest = -1;
				double valBest = -2.0;
                for (int itIdx=0; itIdx<arrUnitPatternSims.Length; itIdx++) {
                    if (actionCode == ctx.units[itIdx].actionCode) { // action must be the same to count as the same
                        if (arrUnitPatternSims[itIdx].sim > valBest) {
                            idxBest = itIdx;
							valBest = arrUnitPatternSims[itIdx].sim;
						}
					}
				}
				
				if (valBest > paramGoodEnoughSimThreshold) {

                    // make sure that the action is the same. I guess this is very important that actions which dont get reward by matching die off at some point.
					if (ctx.units[idxBest].actionCode == actionCode) {
						wasMatchFound = true;
						
						// reward unit
						// TODO LOW
					}
				}
			}
			
			if (!wasMatchFound) {

                // we add a new unit if no match was found

				UnitB createdUnit = UnitPatternUtils.makeUnitByStimulus(lastPerceivedStimulus, actionCode, consequenceVec, predictedReward);
				
				ctx.units.Add(createdUnit);
				// TODO : care about AIKR here
				
				//Console.WriteLine("");
				//Console.WriteLine("DBG :  created new unit by contingency");
			}
		}
	}

	// compute next "expected" observation and reward based on given observation and the action done to the environment
	public Tuple<Vec, double> calcNextObservationStateAndRewardGivenObservationAndAction(Vec observation, string actionCode) {
		VotingWeightsOfUnits votingOfUnits;
		
		// compute similarity of observation to units in ctx
		{
			SimilarityCalculationStrategy similarityCalcStrategy;
			similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy();
			//similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy(); // use attention strategy
			UnitPatternSim[] arrUnitPatternSims = similarityCalcStrategy.calcMatchingScore__by__stimulus(observation, ctx);
			
			Vec vecSim = LearnerUtils.extractSim(arrUnitPatternSims); // extract only similarity values

			if (vecSim.arr.Length == 0) {
				return null;
			}

			votingOfUnits = new VotingWeightsOfUnits( VecUtils.normalize(vecSim) );
		}

		// we only acre about a specific action,    so now we need to mask out the voting for the units with the wrong action
		{
			for (int idx=0; idx<ctx.units.Count; idx++) {
				if (ctx.units[idx].actionCode != actionCode) {
					votingOfUnits.v.arr[idx] = 0.0;
				}
			}
		}

		votingOfUnits = new VotingWeightsOfUnits( VecUtils.normalizeL1( votingOfUnits.v ) ); // need to normalize it for correct weighting

		// now we just need to extrapolate the predicted next state
		Vec predictedNextState = CortialCore.computePredictedVector(votingOfUnits, ctx);
		
		// we also need to interpolate the predicted reward
		double predictedReward = CortialCore.calcWeightedPredictedReward(votingOfUnits.v, ctx);
		
		return new Tuple<Vec, double>(predictedNextState, predictedReward);
	}
}







/*

learner is based on following ideas:

* cortial algorithms as core methodology of substrate

* computing similarity between neurons :  modern hopfield neural networds                reference "Hopfield Networks is All You Need" https://arxiv.org/abs/2008.02217
* unsupervised learning of prediction  :  hierachical temporal memory (HTM) theory       reference "On Intelligence" https://en.wikipedia.org/wiki/On_Intelligence

* TODO : decision making by voting of multiple column: TODO : search paper from hawkins

*/

// context of the cortial algorithm - learning and inference and interaction with the environment
public class CortialAlgoithm_LearnerCtx {
	public ColumnCtxA[] columns = null; // columns similar to cortial columns
	

	// predictive NN to learn to associate obseration,action with effect vector and reward
	public PredictiveNn predictiveNn = new PredictiveNn();
	
	// reward statistics
	public long cntRewardPos = 0;
	public long cntRewardNeg = 0;
	
	
	public double paramRandomActionChance = 0.05; // probability to act with a random action in each timestep
	
	public double paramGoodEnoughSimThreshold = 0.95;
	

	public string lastSelectedActionCode = null;
	public Vec lastPerceivedStimulus = null;
	public int lastReward = 0; // reward from last cycle    - used to learn associations to predict reward


	public RngA rng = new RngA();
	
	
	public EnvAbstract env = null; // must be set externally
	
	public int verbosity = 0;

	public void allocateColumns(int nColumns) {
		columns = new ColumnCtxA[nColumns];
		for(int idx=0;idx<columns.Length;idx++) {
			columns[idx] = new ColumnCtxA();
		}
	}
	
	public void resetColumnStates() {

		for (int idx=0;idx<columns.Length;idx++) {
			columns[idx].lastPerceivedStimulus = null;
			columns[idx].lastSelectedAction = null;
		}
	}
	
	
	// synchronous step between learner and environment
	public void learnerSyncronousAndEnviromentStep(long globalIterationCounter) {
		
		
		


		if (verbosity >= 1) {
			Console.WriteLine("");
			Console.WriteLine("");
			Console.WriteLine("");
		}
		
		// DEBUG units
		if (verbosity >= 10) {
			foreach (ColumnCtxA itColumn in columns) {
				Console.WriteLine("units:");
				foreach (UnitB itUnit in itColumn.ctx.units) {
					Console.WriteLine("");
					Console.WriteLine(UnitUtils.retDebugStrOfUnit(itUnit));
				}
				Console.WriteLine("");
			}
		}
		
		
		env.setGlobalIterationCounter(globalIterationCounter);
		
		Vec perceivedStimulus;
		
		perceivedStimulus = env.receivePerception();
		

		if (verbosity >= 2) {
			Console.WriteLine(string.Format("perceived stimulus= {0}", VecUtils.convToStr(perceivedStimulus)));
		}
		
		
		foreach (ColumnCtxA itColumn in columns) {
			// build contigency from past observation, past action and current stimulus and update in memory
			buildContingencyFromPastObservation(perceivedStimulus, itColumn);
		}

		{
			predictiveNn.learn(lastPerceivedStimulus, perceivedStimulus, lastSelectedActionCode, lastReward);
		}
		
		
		

		
		
		/* commented because DEPRECTAED because new functionality can do this much better!
		
		

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
		
		int nPlanningDepth = 5;
		int nPlanningAttempts = 10;

		AlgorithmResult__Planning resBestPlanning = null; // best planning result with highest future reward
		for(int itPlanningAttempt=0; itPlanningAttempt<nPlanningAttempts; itPlanningAttempt++) {
			AlgorithmResult__Planning resPlanning = CortialCore.LAB__cortialAlgorithm__planning_A(perceivedStimulus, this, nPlanningDepth);
			if (resBestPlanning == null || resPlanning.expectedRewardSum > resBestPlanning.expectedRewardSum) {
				resBestPlanning = resPlanning;
			}
		}


		
		string maxVotingActionCode = resBestPlanning.firstActionActionCode; // copy selected action code into temporary variable
		
		
		
		
		
		// DEBUG
		Console.WriteLine(string.Format("selected max firstActionActionCode={0}    expected future rewardSum={1}", resBestPlanning.firstActionActionCode, resBestPlanning.expectedRewardSum));
		
		string selectedActionCode = null;
		
		if (!(maxVotingActionCode is null)) {
			selectedActionCode = maxVotingActionCode;
		}
		
		
		
		
		
		
		
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
			int idxSel = (int)rng.nextInteger(columns[0].availableActions.Count);
			selectedActionCode = columns[0].availableActions[idxSel];
			Console.WriteLine("DBG  random action is selected");
		}
		
		
		// DEBUG
		Console.WriteLine(string.Format("selected actionCode={0}", selectedActionCode));
		
		// stage: DO ACTUAL ACTION
		{
			env.doAction(selectedActionCode);
		}
		
		int rewardFromLastAction = env.retRewardFromLastAction();
		
		// stage: receive reward from environment
		{
			foreach (ColumnCtxA itColumn in columns) {
				// NOTE that 0 is no reward signal
				itColumn.lastRewardFromEnvironment = rewardFromLastAction;
			
				if (itColumn.lastRewardFromEnvironment < 0) {
					cntRewardNeg+=1;
				}
				else if (itColumn.lastRewardFromEnvironment > 0) {
					cntRewardPos+=1;
				}
			
				if (itColumn.lastRewardFromEnvironment < 0) {
					Console.WriteLine("column: received - NEGATIVE reward!");
				}
				else if (itColumn.lastRewardFromEnvironment > 0) {
					Console.WriteLine("column: received + POSITIVE reward!");
				}

				// TODO : use reward before next cycle to reward/punish units
			}
		}
		
		
		// stage: prepare next cycle
		foreach (ColumnCtxA itColumn in columns) {
			itColumn.lastSelectedAction = selectedActionCode;
			itColumn.lastPerceivedStimulus = perceivedStimulus;
		}

		{
			lastSelectedActionCode = selectedActionCode;
			lastPerceivedStimulus = perceivedStimulus;
			lastReward = rewardFromLastAction;
		}
		
		
		
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
		
		
		foreach (ColumnCtxA itColumn in columns) {
			// build contigency from past observation, past action and current stimulus and update in memory
			buildContingencyFromPastObservation(perceivedStimulus, itColumn);
		}

		{
			predictiveNn.learn(lastPerceivedStimulus, perceivedStimulus, lastSelectedActionCode, lastReward);
		}
	}
	
	
	
	
	
	
	// (private)
	//
	// build contigency from past observation, past action and current stimulus and update in memory
	//
	public void buildContingencyFromPastObservation(Vec perceivedStimulus, ColumnCtxA column) {
		// NOTE : we only update if it is similar enough to existing condition
		
		if (!(column.lastPerceivedStimulus is null)) {
			
			bool wasMatchFound = false;
			
			// search for match
			SimilarityCalculationStrategy similarityCalcStrategy = new SoftMaxSimilarityAttentionCalculationStrategy();
			UnitPatternSim[] arrUnitPatternSims = similarityCalcStrategy.calcMatchingScore__by__stimulus(column.lastPerceivedStimulus, column.ctx);
			
			if (verbosity >= 10) {
				// DEBUG
				Console.WriteLine("");
				Console.WriteLine("sim to perceived contigency:");

				Vec vecSim = LearnerUtils.extractSim(arrUnitPatternSims);

				Console.WriteLine(VecUtils.convToStr(vecSim));
			}
			
			
			
			// decide if the match is good enough
			{
				int idxBest = -1;
				double valBest = -2.0;
				for (int itIdx=0; itIdx<arrUnitPatternSims.Length; itIdx++) {
					if (column.lastSelectedAction == column.ctx.units[itIdx].actionCode) { // action must be the same to count as the same
						if (arrUnitPatternSims[itIdx].sim > valBest) {
							idxBest = itIdx;
							valBest = arrUnitPatternSims[itIdx].sim;
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

				UnitB createdUnit = UnitPatternUtils.makeUnitByStimulus(column.lastPerceivedStimulus, column.lastSelectedAction, perceivedStimulus, column.lastRewardFromEnvironment);
								
				column.ctx.units.Add(createdUnit);
				// TODO : care about AIKR here
				
				Console.WriteLine("");
				Console.WriteLine("DBG :  created new unit by contingency");
			}
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











// lab idea: use multiplication mask as attention

// idea: use multiplication mask as attention. this form of attention allows the units to mask out certain channels



//	double[] function(Vec perceivedStimulus, CtxZZZ ctx)  calcSimFn;
//	calcSimFn = &calcSimArrByAttention;
//	
//	double[] arrSim = calcSimFn(null, null);

// tuple of similarity and actual unit pattern and unit to which the pattern belongs to
public class UnitPatternSim {
	public UnitB unit;
	public UnitPattern unitPattern;
	public double sim; // similarity from 0.0 to 1.0

	public UnitPatternSim(UnitB unit, UnitPattern unitPattern, double sim) {
        this.unit = unit;
        this.unitPattern = unitPattern;
        this.sim = sim;
    }
}

public class SoftMaxSimilarityAttentionCalculationStrategy : SimilarityCalculationStrategy {
	public override UnitPatternSim[] calcMatchingScore__by__stimulus(Vec stimulus, CtxZZZ ctx) {
		return calcSimArrByAttention(stimulus, ctx);
	}

	private static UnitPatternSim[] calcSimArrByAttention(Vec perceivedStimulus, CtxZZZ ctx) {
		List<UnitPatternSim> listUnitPatternSim = new List<UnitPatternSim>();
		
		////double[] arrUnitSim = new double[ctx.units.Count];
	

		foreach (UnitB itUnit in ctx.units) {
			
			foreach (UnitPattern itUnitPattern in itUnit.patterns) {
				Vec postAttention;
				if (true) { // use attention?
					postAttention = VecUtils.mulComponents(perceivedStimulus, itUnitPattern.attentionMask);
				}
				else {
					// COMMENTED BECAUSE NOT TRIED
		
					// else post attention is just perceivedStimulus
					postAttention = perceivedStimulus;
				}

				Vec key = itUnit.v; // key vector which we take from the pattern for which the unit is looking for
		
				double voteSimScalar = VecUtils.calcDot(key, postAttention);
		
				listUnitPatternSim.Add( new UnitPatternSim(itUnit, itUnitPattern, voteSimScalar) );
			}
		}
		
		////return arrUnitSim;
		return listUnitPatternSim.ToArray();
	}
}




// learning algorithm next step:
//    reward attention pattern of the winner unit and punish attention pattern of the looser units
//    TODO TODO TODO



// TODO LOW : attention: use attention and also implement attentionMask rewarding + punishment to allow the algorithm to update the attention masks
//    TODO : use attention in main loop
//    TODO : implement update of attention masks of winner unit and looser units of voting






// DONE : use "predictedReward" for action selection based on stimuli!























// Jeff Hawkin : "reference frames"  : explained in the book "The 1000 brains theory"









public struct Vec2i {
    public long x;
    public long y;

    public Vec2i(long x, long y) {
        this.x = x;
        this.y = y;
    }

	
	public static Vec2i operator +(Vec2i a, Vec2i b) => new Vec2i(a.x+b.x, a.y+b.y);
	public static Vec2i operator -(Vec2i a, Vec2i b) => new Vec2i(a.x-b.x, a.y-b.y);
}


public class Vec2iUtils {
    public static Vec2i sub(Vec2i lhs, Vec2i rhs) {
	    return new Vec2i(lhs.x-rhs.x, lhs.y-rhs.y);
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

	// make it to sum to 1.0
	public static Vec normalizeL1(Vec v) {
		double l = 0.0;
		foreach (double iv in v.arr) {
			l += iv;
		}

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
		double rngReal = nextReal();
		long v = (long)(rngReal * max * 1000) % max;

		return v;
	}
}







// TODO  :  put actions for decision making into global context of learner!

