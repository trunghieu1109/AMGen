async def forward_168(self, taskInfo):
    logs = []

    cot_sc_instruction0 = (
        "Sub-task 0: Apply the transformation from the original decay 2A -> 2B + 2E + 2V to the variant decay 2A -> 2B + 2E + M, "
        "identifying changes in emitted particles and their properties with context from the query."
    )
    final_decision_instruction0 = (
        "Sub-task 0: Synthesize and choose the most consistent understanding of the particle transformation in the decay variant."
    )
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'final_decision_instruction': final_decision_instruction0,
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results0, log0 = await self.sc_cot(
        subtask_id="subtask_0",
        cot_agent_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    logs.append(log0)

    debate_instruction1 = (
        "Sub-task 1: Analyze the kinematic and energy conservation implications of replacing two V particles with one massless M particle on the total energy spectrum of the E particles, "
        "focusing on spectrum continuity and endpoint changes, using the output from Sub-task 0."
    )
    final_decision_instruction1 = (
        "Sub-task 1: Debate and conclude the effects on the E particle energy spectrum's continuity and endpoint due to the particle replacement."
    )
    debate_desc1 = {
        'instruction': debate_instruction1,
        'final_decision_instruction': final_decision_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'context_desc': ["user query", "thinking of subtask 0", "answer of subtask 0"],
        'temperature': 0.5
    }
    results1, log1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        n_repeat=self.max_round
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Integrate physical insights to determine how the number and mass of emitted particles affect the degrees of freedom and energy partitioning, "
        "thereby influencing whether the E particle spectrum remains continuous or becomes discrete, based on Sub-task 0 output."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and select the most consistent conclusion on the spectrum nature and energy partitioning."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'final_decision_instruction': final_decision_instruction2,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 0", "answer of subtask 0"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Evaluate the analyzed spectrum characteristics from Sub-tasks 1 and 2 against the provided answer choices to select the option that correctly describes the changes in the E particle energy spectrum in the variant decay."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Debate and finalize the correct answer choice describing the spectrum changes in the variant decay."
    )
    debate_desc3 = {
        'instruction': debate_instruction3,
        'final_decision_instruction': final_decision_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'context_desc': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"],
        'temperature': 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'])
    return final_answer, logs
