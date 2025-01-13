

import std.stdio;
import std.string : join;
import std.format : format;

import std.datetime;


// for GUID generation
import std.uuid;







import mlA;

// for testing
import mlB;

import mlC;



void manualtest__aeraish_A() {
    ControlA control = new ControlA();

    // register ops
    {
        OpTestA createdOp = new OpTestA();
        control.opRegistry.registerOp("^testA", createdOp);
    }

    // add goals for testing
    {
        AbstractGoalA createdGoal = new RealizeOpGoalA( new OpInvokeTermItem("^testA") );
        mem_putGoal(control, createdGoal);
    }


    {
        ConfigExternalDeviceArg[] configurationArgs = [];
        configurationArgs ~= new ConfigExternalDeviceArg(3);
        ConfigExternalDeviceTermA createdSentenceTerm = new ConfigExternalDeviceTermA("devA", configurationArgs);
        EventA createdEvent = new TimepointEvent(createdSentenceTerm);
        eventHappened(control, createdEvent);
    }

    {
        TermRefedItem createdSentenceTerm = new OpInvokeTermItem("^testB");
        EventA createdEvent = new SentenceEvent(createdSentenceTerm);
        eventHappened(control, createdEvent);
    }

    {
        TermRefedItem createdSentenceTerm = new NamedTermATermItem("X");
        EventA createdEvent = new SentenceEvent(createdSentenceTerm);
        eventHappened(control, createdEvent);
    }




    for(long z0 = 0; z0 < 1; z0++) {

        SysTime start = Clock.currTime(); // Record the start time

        {
            ProcessMeasurementTimepointItem timepointItem = new ProcessMeasurementTimepointItem("timespanA_0", EnumProcessMeasurementType.START, start);
            EventA createdEvent = new TimepointEvent(timepointItem);
            eventHappened(control, createdEvent);
        }

        {
            // here we are doing the compute task
            // BEGIN COMPUTATION TASK

            QQQ_computeMlExp0();
			

			QQQ_NN029982();
			

            // END COMPUTATION TASK


            TermRefedItem createdSentenceTerm = new OpInvokeTermItem("^testC");
            EventA createdEvent = new SentenceEvent(createdSentenceTerm);
            eventHappened(control, createdEvent);
        }




        {
            SysTime stop = Clock.currTime();   // Record the end time

            ProcessMeasurementTimepointItem timepointItem = new ProcessMeasurementTimepointItem("timespanA_0", EnumProcessMeasurementType.STOP, stop);
            EventA createdEvent = new TimepointEvent(timepointItem);
            eventHappened(control, createdEvent);
        }

        // add event which gives information about inference time some process took
        {
            SysTime end = Clock.currTime();   // Record the end time
            Duration elapsed = end - start;
        
            double dtSeconds = cast(double)elapsed.total!"nsecs" / 1.0e9;

            //dtSeconds = 0.0001; // dummy value for testing

            writeln("Elapsed time: ", dtSeconds, " seconds");
            

            TermNamedItem[] statisticsArgs = [];
            //statisticsArgs ~= new TermNamedItem(new NamedTermATermItem("inferenceTimeA"), new RealVariant(dtSeconds));
            statisticsArgs ~= new TermNamedItem(new NamedTermATermItem("refTimepointName"), new StringVariant("timespanA_0"));
            InternalStatisticsRefedItem createdSentenceTerm = new InternalStatisticsRefedItem("internalStatA", statisticsArgs);
            EventA createdEvent = new TimepointEvent(createdSentenceTerm);
            eventHappened(control, createdEvent);
        }

        {
            TermRefedItem createdSentenceTerm = new NamedTermATermItem("Y");
            EventA createdEvent = new SentenceEvent(createdSentenceTerm);
            eventHappened(control, createdEvent);
        }


    }





    long iterationCounter = 0;
    for(;;) {
        if (iterationCounter > 5) {
            break;
        }

        writeln("");
        writeln("");
        writeln("iteration=", iterationCounter);
        writeln("");

        iterationCounter++;


        {
            AbstractGoalA goalWorking = mem_selectRemoveGoalWithHighestPriority(control);

            if (!(goalWorking is null)) {
                // process goal

                {
                    RealizeOpGoalA realizedOpGoal = cast(RealizeOpGoalA)goalWorking;
                    TermGoalA termGoal = cast(TermGoalA)goalWorking;
                    if (realizedOpGoal !is null) {
                        writeln(format("RealizeOpGoalA %s", realizedOpGoal.termInvoke.convToStr()));

                        // search in op-registry and then invoke
                        OpA op = control.opRegistry.lookupByName(realizedOpGoal.termInvoke.opName);
                        if (!(op is null)) {
                            op.invoke();

                            // this is a event which happened, so we keep track of it
                            EventA createdEvent = new SentenceEvent(realizedOpGoal.termInvoke);
                            eventHappened(control, createdEvent);
                        }
                    }
                    else if (termGoal !is null) {
                        writeln(format("TermGoalA %s", termGoal.term.convToStr()));

                        // derive goals based on beliefs using applied rules
                        
                        // TODO TODO TODO TODO
                    }
                    else {
                        // not implemented path
                        // we silently ignore this
                    }
                }
            }
        }
    }

    // manual test: force analysis
    /*{
        writeln("");

        PredImplTermItem[] analysisResult_predImpls = trace__queryPredImpls(control.trace, &checkIsNotOpInvokeTerm);

        writeln(format("n=%d", analysisResult_predImpls.length));

        foreach (itPredImpl; analysisResult_predImpls) {
            string s = convToNalLinkStr(itPredImpl);
            writeln(s);
        }

        writeln("");


        // create sentence for transfering observation as beliefs
        foreach (itPredImpl; analysisResult_predImpls) {
            PredImplSentenceA createdSentence = new PredImplSentenceA(itPredImpl);
            mem_putSentence(control, createdSentence);
        }
    }*/

    // manual test: force analysis of timespan
    {
        writeln("");

        TemporalSpanData[] arrTemporalSpanData = trace__queryTimespanStatisticEvents(control.trace);

        writeln(format("n=%d", arrTemporalSpanData.length));

        foreach (itTemporalSpanData; arrTemporalSpanData) {
            string str = conv_TemporalSpanData_ToStr(itTemporalSpanData);
            writeln("");
            writeln(str);
        }

    }
}






// core class of the control
class ControlA {

    // memory
    public PredImplSentenceA[] sentences;

    // memory
    public AbstractGoalA[] goals;

    public OpRegistry opRegistry = new OpRegistry();


    public TraceA trace = new TraceA();


    public final this() {
    }
}

// cares about revision
void mem_putSentence(ControlA self, PredImplSentenceA sentence) {
    
    bool didReviseAny = false;

    foreach (itBeliefSentence; self.sentences) {
        if (checkTermSame(sentence.term, itBeliefSentence.term)) {

            // revision - add evidence
            sentence.tv = NalTv.revision(sentence.tv, itBeliefSentence.tv);
            didReviseAny = true;

            break;
        }
    }

    if (!didReviseAny) {
        // we didn't revise when we are here, so we add a new sentence

        // TODO : take care of AIKR
        self.sentences ~= sentence;
    }
}

void mem_putGoal(ControlA self, AbstractGoalA goal) {
    // TODO : take care of AIKR
    self.goals ~= goal;
}

// return NULL if no goal was found
// select goal with highest priority and remove
AbstractGoalA mem_selectRemoveGoalWithHighestPriority(ControlA self) {
    if (self.goals.length == 0) {
        return null;
    }

    long goalIdxWithHighestPriority = 0;
    double highhestPriority = self.goals[0].priority;

    foreach (itIdx, itGoal; self.goals) {
        if (itGoal.priority > highhestPriority) {
            highhestPriority = itGoal.priority;
            goalIdxWithHighestPriority = itIdx;
        }
    }

    AbstractGoalA resGoal = self.goals[goalIdxWithHighestPriority];

    // remove (slow algorithm)
    AbstractGoalA[] goals2 = [];
    foreach (itIdx, itGoal; self.goals) {
        if (itIdx != goalIdxWithHighestPriority) {
            goals2 ~= itGoal;
        }
    }
    self.goals = goals2;


    return resGoal;
}



class OpRegistry {
    public OpRegistryItem[] items;

    public void registerOp(string opName, OpA op) {
        items ~= new OpRegistryItem(opName, op);
    }

    public OpA lookupByName(string opName) {
        foreach (itOp; items) {
            if (itOp.opName == opName) {
                return itOp.op;
            }
        }

        return null; // nothing found -> return null
    }
}

class OpRegistryItem {
    public string opName;
    public OpA op;

    public final this(string opName, OpA op) {
        this.opName = opName;
        this.op = op;
    }
}








// predictive implication which 
class PredImplSentenceA {
    //public PredImplTermA term;
    public PredImplTermItem term;

    public NalTv tv = new NalTv(1.0, 1.0);

    public final this(PredImplTermItem term) {
        this.term = term;
    }
}



string convToNalLinkStr(TermRefedItem arg) {
    // TODO LOW: implement conversation of TV!

    return arg.convToStr();
}




abstract class AbstractGoalA {
    public double priority = 0.99999;
}

// goal to realize a op
class RealizeOpGoalA : AbstractGoalA {
    public OpInvokeTermItem termInvoke; // term to invoke the goal

    public final this(OpInvokeTermItem termInvoke) {
        this.termInvoke = termInvoke;
    }
}

class TermGoalA : AbstractGoalA {
    public TermRefedItem term;

    public final this(TermRefedItem term) {
        this.term = term;
    }
}




// Operator
abstract class OpA {
    public abstract void invoke();
}

// op for testing
class OpTestA : OpA {
    public override void invoke() {
        writeln("op OpTestA has been invoked");
    }
}







// idea: log input events and done ops to a trace to be able to analyze for 
class TraceA {
    // events of the trace
    public EventA[] traceEvents;

    long idxSliceStart = 0; // start of the range for the readout of new events to build new contigencies

    public void appendEvent(EventA event) {
        // TODO LOW: care about AIKR

        traceEvents ~= event;
    }
}





abstract class EventA {
}

class SentenceEvent : EventA {
    public TermRefedItem term; // term of the sentence

    public final this(TermRefedItem term) {
        this.term = term;
    }
}

class TimepointEvent : EventA {
    public TimepointItem data;

    public final this(TimepointItem data) {
        this.data = data;
    }
}

/*
class EventA {
    public TermRefedItem term; // term of the sentence

    public final this(TermRefedItem term) {
        this.term = term;
    }
}
*/
// idea: 
// event can be of type 
// * INPUT for a input event 
// * INTERNAL for a internal event
// * REFLECTION for events about self-reflection


// called whenever a event happened
void eventHappened(ControlA self, EventA event) {
    // we need to log the event
    self.trace.appendEvent(event);
}

// called to analyze the trace
//
// tries to build predImpl's where a op call term is sandwiched.
PredImplTermItem[] trace__queryPredImpls(TraceA trace, bool function(TermRefedItem term) fnFilterCondition ) {
    PredImplTermItem[] predImplTerms;

    for (long idx=trace.idxSliceStart+2; idx<trace.traceEvents.length; idx++) {
        
        SentenceEvent sentenceEventIdxMinus0 = cast(SentenceEvent)trace.traceEvents[idx-0];
        SentenceEvent sentenceEventIdxMinus1 = cast(SentenceEvent)trace.traceEvents[idx-1];
        SentenceEvent sentenceEventIdxMinus2 = cast(SentenceEvent)trace.traceEvents[idx-2];
        if (sentenceEventIdxMinus0 !is null && sentenceEventIdxMinus1 !is null && sentenceEventIdxMinus2 !is null) {
            if (cast(OpInvokeTermItem)sentenceEventIdxMinus1.term !is null) {

                TermRefedItem conditionTerm = sentenceEventIdxMinus2.term;
                if (fnFilterCondition(conditionTerm)) { // condition term must match with the actual query
                    OpInvokeTermItem opInvokeTerm = cast(OpInvokeTermItem)sentenceEventIdxMinus1.term;
                    TermRefedItem consequenceTerm = sentenceEventIdxMinus0.term;

                    PredImplTermItem predImplTerm = new PredImplTermItem(new SequenceTermItem([conditionTerm, opInvokeTerm]), consequenceTerm);

                    predImplTerms ~= predImplTerm;
                }
            
            }
        }
    }

    trace.idxSliceStart = trace.traceEvents.length; // we want that the next readout starts fresh

    return predImplTerms;
}





class NalTv {
    public double evidencePos = 0.0;
    public double evidenceTotal = 0.0;

    public this(double evidencePos, double evidenceTotal) {
        this.evidencePos = evidencePos;
        this.evidenceTotal = evidenceTotal;
    }

    public double calcFreq() {
        return evidencePos / evidenceTotal;
    }

    public static NalTv revision(NalTv a, NalTv b) {
        return new NalTv(a.evidencePos + b.evidencePos, a.evidenceTotal + b.evidenceTotal);
    }
}




// TODO MID : implement ML translation code to analyze trace of events to datamine for useful stuff for meta-control
//    TODO : convert read out sequence to NN friendly encoding by encoding the symbol names as vectors, etc.










// AERA inspired

// reference to some term - commented because not necessary
//class TermRef {
//    public TermRefedItem target;
//}

// thing which is referenced and can be payload of a event
//abstract class RefedItem {
//    public abstract string convToStr();
//}

abstract class TimepointItem {
    public abstract string convToStr();
}

// actual term
abstract class TermRefedItem {
    public abstract string convToStr();
}

// commented because not used
bool checkTermSame(TermRefedItem a, TermRefedItem b) {
    if (cast(NamedTermATermItem)a !is null && cast(NamedTermATermItem)b !is null) {
        NamedTermATermItem a2 = cast(NamedTermATermItem)a;
        NamedTermATermItem b2 = cast(NamedTermATermItem)b;

        return a2.name == b2.name;
    }
    if (cast(PredImplTermItem)a !is null && cast(PredImplTermItem)b !is null) {
        PredImplTermItem a2 = cast(PredImplTermItem)a;
        PredImplTermItem b2 = cast(PredImplTermItem)b;
        return checkTermSame(a2.subj, b2.subj) && checkTermSame(a2.pred, b2.pred);
    }
    else if (cast(SequenceTermItem)a !is null && cast(SequenceTermItem)b !is null) {
        SequenceTermItem a2 = cast(SequenceTermItem)a;
        SequenceTermItem b2 = cast(SequenceTermItem)b;

        if (a2.seqItems.length != b2.seqItems.length) {
            return false;
        }

        for (long idx=0;idx<a2.seqItems.length;idx++) {
            if (!checkTermSame(a2.seqItems[idx], b2.seqItems[idx])) {
                return false;
            }
        }

        return true;
    }
    else if (cast(OpInvokeTermItem)a !is null && cast(OpInvokeTermItem)b !is null) {
        OpInvokeTermItem a2 = cast(OpInvokeTermItem)a;
        OpInvokeTermItem b2 = cast(OpInvokeTermItem)b;

        return a2.opName == b2.opName;
    }
    else if (cast(ConfigExternalDeviceTermA)a !is null && cast(ConfigExternalDeviceTermA)b !is null) {
        ConfigExternalDeviceTermA a2 = cast(ConfigExternalDeviceTermA)a;
        ConfigExternalDeviceTermA b2 = cast(ConfigExternalDeviceTermA)b;

        if (a2.deviceName != b2.deviceName) {
            return false;
        }

        if (a2.configurationArgs.length != b2.configurationArgs.length) {
            return false;
        }

        for (long idx=0;idx<a2.configurationArgs.length;idx++) {
            if (a2.configurationArgs[idx].value != b2.configurationArgs[idx].value) {
                return false;
            }
        }

        return true;
    }

    // TODO : implement code for SubjViewTerm thingy

    return false; // can't be the same if they have different type
}

// view of only the subject of a PredImplTermItem
class PredImplViewOnlySubjItem : TermRefedItem {
    public PredImplTermItem target;

    public this(PredImplTermItem target) {
        this.target = target;
    }

    public override string convToStr() {
        return dereferenceView.convToStr();
    }

    public final TermRefedItem dereferenceView() {
        return target.subj;
    }
}

class NamedTermATermItem : TermRefedItem {
    public string name;

    public this(string name) {
        this.name = name;
    }

    public override string convToStr() {
        return name;
    }
}

// term to invoke a op
class OpInvokeTermItem : TermRefedItem {
    public string opName; // name of op with "^" letter

    public final this(string opName) {
        this.opName = opName;
    }

    public override string convToStr() {
        return format("INVOKE( %s )", opName);
    }
}

// similar to predictive implication like in NAL
class PredImplTermItem : TermRefedItem {
    public TermRefedItem subj;
    public TermRefedItem pred;

    public this(TermRefedItem subj, TermRefedItem pred) {
        this.subj = subj;
        this.pred = pred;
    }

    public override string convToStr() {
        string strSubj = subj.convToStr();
        string strPred = pred.convToStr();
        return format("%s =/> %s", strSubj, strPred);
    }
}

class SequenceTermItem : TermRefedItem {
    public TermRefedItem[] seqItems;

    public final this(TermRefedItem[] seqItems) {
        this.seqItems = seqItems;
    }

    public override string convToStr() {
        string[] strsOfItem = [];
        foreach (itSeqItem; seqItems) {
            strsOfItem ~= itSeqItem.convToStr();
        }
        string strItems = strsOfItem.join(", ");
        return format("( %s )", strItems);
    }
}







// term which is used for configuration of a external device
class ConfigExternalDeviceTermA : TimepointItem {
    public string deviceName; // name of the device to be configured

    // arguments for the configuration of the 
    public ConfigExternalDeviceArg[] configurationArgs;

    public final this(string deviceName, ConfigExternalDeviceArg[] configurationArgs) {
        this.deviceName = deviceName;
        this.configurationArgs = configurationArgs;
    }

    public override string convToStr() {
        string[] strOfArg;
        foreach (itArg; configurationArgs) {
            strOfArg ~= format("%d", itArg.value);
        }

        string strOfArgs = strOfArg.join(",");

        return format("CONFIG( dev=%s args=<%s>)", deviceName, strOfArgs);
    }
}

class ConfigExternalDeviceArg {
    //public string key; // key is not necessary because it's implicitly contained in the index of the argument
    public long value;

    public final this(long value) {
        //this.key = key;
        this.value = value;
    }
}

// TODO : ConfigExternalDeviceTermA :: also implement ops for manipulating external devices







// statistics of something internal
class InternalStatisticsRefedItem : TimepointItem {
    public string sensorName; // name of the internal sensor
    public TermNamedItem[] namedItems; // actual items of the values which are associated with the sensor values

    public final this(string sensorName, TermNamedItem[] namedItems) {
        this.sensorName = sensorName;
        this.namedItems = namedItems;
    }

    public TermNamedItem lookupItemByNameTerm(NamedTermATermItem nameTerm) {
        foreach (itItem; namedItems) {
            if (checkTermSame(nameTerm, itItem.nameTerm)) {
                return itItem;
            }
        }

        return null; // nothing found -> we return null in this case
    }

    public override string convToStr() {
        string[] strsNamedItem = [];
        foreach (itNamedItem; namedItems) {
            // TODO
            //strsNamedItem ~= format("%s=%f", itNamedItem.nameTerm.convToStr(), itNamedItem.valueReal);
        }
        string namedItems = strsNamedItem.join(", ");
        return format("(INTERNAL_STATISTICS %s)", namedItems);
    }
}

class TermNamedItem {
    public NamedTermATermItem nameTerm; // name of the item as a term
    public Variant value;

    public final this(NamedTermATermItem nameTerm, Variant value) {
        this.nameTerm = nameTerm;
        this.value = value;
    }
}






// type: TimepointItem
class ProcessMeasurementTimepointItem : TimepointItem {
    public string timepointName; // unique name of the timepoint

    public EnumProcessMeasurementType type;

    public SysTime absoluteTime;

    public this(string timepointName, EnumProcessMeasurementType type, SysTime absoluteTime) {
        this.timepointName = timepointName;
        this.absoluteTime = absoluteTime;
        this.type = type;
    }

    public override string convToStr() {
        string strType = "START";
        if (type == EnumProcessMeasurementType.STOP) {
            strType = "STOP";
        }
        return format("TIMEPOINT(uniqueName?=%s, type=%s)", timepointName, strType);
    }
}

enum EnumProcessMeasurementType {
    START,
    STOP
}




// use as a delegate function to filter the candidate predImpl from the trace
bool checkIsInternalStatisticsRefedItem(TermRefedItem arg) {
    return cast(InternalStatisticsRefedItem)arg !is null;
}

// use as a delegate function to filter the candidate predImpl from the trace
bool checkIsNotOpInvokeTerm(TermRefedItem arg) {
    return cast(OpInvokeTermItem)arg is null;
}




// TODO MID : make use of InternalStatisticsRefedItem to measure inference time of a restricted sub-process caused by NAL-9
//     TODO : measure execution time which a process takes
//     TODO : do decision making based on that





// scratchpad information to hold temporary stored timespan between events and associated meta-data
class TemporalSpanData {
    public ProcessMeasurementTimepointItem timepointEventStart;
    public ProcessMeasurementTimepointItem timepointEventStop;
    public InternalStatisticsRefedItem statItem;

    public final this(ProcessMeasurementTimepointItem timepointEventStart, ProcessMeasurementTimepointItem timepointEventStop, InternalStatisticsRefedItem statItem) {
        this.timepointEventStart = timepointEventStart;
        this.timepointEventStop = timepointEventStop;
        this.statItem = statItem;
    }
}

// helper to convert to string for debugging
string conv_TemporalSpanData_ToStr(TemporalSpanData arg) {
    string res = "";

    res ~= format("sensorName=%s", arg.statItem.sensorName);

    TermNamedItem datItem = arg.statItem.lookupItemByNameTerm(new NamedTermATermItem("refTimepointName"));
    if (datItem !is null) {
        if (cast(StringVariant)datItem.value !is null) {
            string strValue = (cast(StringVariant)datItem.value).value;
            res ~= "\n" ~ format("   refTimepointName=%s", strValue);
        }
    }

    return res;
}


// code which scans trace for START_ ... STOP_ internalStat(timespan) which has the same id as start and stop
TemporalSpanData[] trace__queryTimespanStatisticEvents(TraceA trace) {

    TemporalSpanData[] arrResult = [];

    // scan for start
    for(long idxStart=0; idxStart < trace.traceEvents.length; idxStart++) {

        TimepointEvent timepointEventStart = cast(TimepointEvent)trace.traceEvents[idxStart];
        if (timepointEventStart !is null) {

            ProcessMeasurementTimepointItem startTimepointItem = cast(ProcessMeasurementTimepointItem)timepointEventStart.data;
            if (startTimepointItem !is null && startTimepointItem.type == EnumProcessMeasurementType.START) {
                

                // scan for same stop
                for(long idxStop=idxStart+1;idxStop<trace.traceEvents.length; idxStop++) {
                    TimepointEvent timepointEventStop = cast(TimepointEvent)trace.traceEvents[idxStop];
                    if (timepointEventStop !is null) {

                        ProcessMeasurementTimepointItem stopTimepointItem = cast(ProcessMeasurementTimepointItem)timepointEventStop.data;
                        if (stopTimepointItem !is null && stopTimepointItem.type == EnumProcessMeasurementType.STOP && startTimepointItem.timepointName == stopTimepointItem.timepointName) {
                            // start+stop timepoint events are found

                            //writeln("start+stop"); // DBG


                            
                            // scan for internal statistics which links back to information about the two timepoints

                            for (long idxStatItem=idxStop+1;idxStatItem<trace.traceEvents.length;idxStatItem++) {
                                TimepointEvent timepointEventForStat = cast(TimepointEvent)trace.traceEvents[idxStatItem];
                                if (timepointEventForStat !is null) {


                                    InternalStatisticsRefedItem statsReffedItem = cast(InternalStatisticsRefedItem)timepointEventForStat.data;
                                    
                                    
                                    // it also has to link back to the ProcessMeasurementTimepointItem
                                    TermNamedItem datItem = statsReffedItem.lookupItemByNameTerm(new NamedTermATermItem("refTimepointName"));
                                    if (datItem !is null) {
                                        if (cast(StringVariant)datItem.value !is null && (cast(StringVariant)datItem.value).value == startTimepointItem.timepointName) {

                                            // we did find a ProcessMeasurementTimepointItem which refers to the timespan between the two start+stop events.
                                            // 
                                            // now we only need to extract the information

                                            TemporalSpanData createdTemporalSpanData = new TemporalSpanData(startTimepointItem, stopTimepointItem, statsReffedItem);
                                            arrResult ~= createdTemporalSpanData;


                                            goto skip;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        skip:
    }

    return arrResult;
}

abstract class Variant {
}

class RealVariant : Variant {
    public double value;
    public final this(double value) {
        this.value = value;
    }
}

class StringVariant : Variant {
    public string value;
    public final this(string value) {
        this.value = value;
    }
}





















// basic ML experiment
void QQQ_computeMlExp0() {
    ClassifierA classifier = new ClassifierA();


    RngA rng = new RngA();

    Vec[] alphabet;


    {
        long basicVecSize = 12;

        long nSymbols = 20;

        long windowSize = 12;


        for(long itSymbol=0;itSymbol<nSymbols;itSymbol++) {
            alphabet ~= new Vec(genRngVec(basicVecSize, rng));
        }



        long[] tokenSymbols = [0, 0, 1, 6];

        // fill with null-tokens 
        while (tokenSymbols.length < windowSize) {
            tokenSymbols = tokenSymbols ~ (nSymbols-1);
        }

        // convert to real valued vector
        Vec temp0 = new Vec([]);
        foreach (itTokenSymbol; tokenSymbols) {
            temp0 = append(temp0, alphabet[itTokenSymbol]);
        }

        long lenVotingSize = 12; // length of the vector to vote on

        // append vector on which it can vote on the result
        temp0 = append(temp0, makeVecByLength(lenVotingSize));




        Vec vecInput = temp0;

        // add dummy trainingset to NN
        classifier.items ~= new DatItem(vecInput, randomUUID().toString(), 0);
        

        Vec vecOutput = classifier.inference(vecInput);



        // extract required information from vecOutput
        Vec v2 = new Vec(vecOutput.arr[$-lenVotingSize..$]);

        // search highest index in v2, which is our output predicted symbol
        int symbolOutputPredicted = calcHighestValueIdx(v2);

        writeln(format("symbolOutputPredicted=%d", symbolOutputPredicted));


    }

}


