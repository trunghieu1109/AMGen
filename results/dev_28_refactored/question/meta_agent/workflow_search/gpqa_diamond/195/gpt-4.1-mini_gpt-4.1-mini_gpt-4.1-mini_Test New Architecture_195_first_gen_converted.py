async def forward_195(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction0_1 = (
        "Sub-task 1: Extract and consolidate all given information about the system, parameters, "
        "and candidate formulas into a clear summary."
    )
    cot_agent_desc0_1 = {
        "instruction": cot_instruction0_1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results0_1, log0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc0_1,
        n_repeat=self.max_sc
    )
    logs.append(log0_1)

    debate_instruction1_1 = (
        "Sub-task 1: Analyze the physical relationships and constraints governing the relativistic harmonic oscillator, "
        "including Hooke's law, energy relations, and relativistic velocity limits, "
        "using the summary from Stage 0."
    )
    debate_desc1_1 = {
        "instruction": debate_instruction1_1,
        "final_decision_instruction": "Sub-task 1: Analyze physical relationships and constraints of the relativistic harmonic oscillator.",
        "input": [taskInfo, results0_1['thinking'], results0_1['answer']],
        "context_desc": ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"],
        "temperature": 0.5
    }
    results1_1, log1_1 = await self.debate(
        subtask_id="stage_1.subtask_1",
        debate_desc=debate_desc1_1,
        n_repeat=self.max_round
    )
    logs.append(log1_1)

    cot_instruction2_1 = (
        "Sub-task 1: Derive the quantitative expression for the maximum speed v_max of the mass by applying relativistic energy "
        "and momentum relations to the harmonic oscillator system, using outputs from Stage 0 and Stage 1."
    )
    final_decision_instruction2_1 = (
        "Sub-task 1: Synthesize and choose the most consistent derived expression for v_max."
    )
    cot_sc_desc2_1 = {
        "instruction": cot_instruction2_1,
        "final_decision_instruction": final_decision_instruction2_1,
        "input": [taskInfo, results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']],
        "temperature": 0.5,
        "context_desc": [
            "user query",
            "thinking of stage_0.subtask_1",
            "answer of stage_0.subtask_1",
            "thinking of stage_1.subtask_1",
            "answer of stage_1.subtask_1"
        ]
    }
    results2_1, log2_1 = await self.sc_cot(
        subtask_id="stage_2.subtask_1",
        cot_agent_desc=cot_sc_desc2_1,
        n_repeat=self.max_sc
    )
    logs.append(log2_1)

    debate_instruction3_1 = (
        "Sub-task 1: Evaluate the provided candidate formulas for v_max against the derived expression from Stage 2, "
        "to determine which formula correctly represents the maximum speed of the relativistic harmonic oscillator."
    )
    final_decision_instruction3_1 = (
        "Sub-task 1: Determine the correct candidate formula for v_max based on the derivation and analysis."
    )
    debate_desc3_1 = {
        "instruction": debate_instruction3_1,
        "final_decision_instruction": final_decision_instruction3_1,
        "input": [taskInfo, results2_1['thinking'], results2_1['answer']],
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results3_1, log3_1 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc3_1,
        n_repeat=self.max_round
    )
    logs.append(log3_1)

    final_answer = await self.make_final_answer(results3_1['thinking'], results3_1['answer'])
    return final_answer, logs
