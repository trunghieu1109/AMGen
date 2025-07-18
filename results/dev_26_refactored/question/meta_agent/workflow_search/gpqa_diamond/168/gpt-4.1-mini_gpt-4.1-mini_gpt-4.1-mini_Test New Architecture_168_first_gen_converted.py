async def forward_168(self, taskInfo):
    logs = []

    cot_sc_instruction0 = (
        "Sub-task 0_1: Extract and transform the given decay information into a clear, structured representation "
        "of the original and variant decay processes, including particle types, masses, and known spectral features, "
        "with context from the user query."
    )
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'final_decision_instruction': "Sub-task 0_1: Synthesize and choose the most consistent structured representation.",
        'input': [taskInfo],
        'temperature': 0.5,
        'context_desc': ["user query"]
    }
    results0, log0 = await self.sc_cot(
        subtask_id="subtask_0_1",
        cot_agent_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    logs.append(log0)

    debate_instruction1_1 = (
        "Sub-task 1_1: Debate the kinematic and energy conservation implications of replacing 2V particles "
        "with a single massless M particle on the energy distribution of the E particles, focusing on spectrum continuity and endpoint changes, "
        "using the structured decay information from Sub-task 0_1."
    )
    debate_desc1_1 = {
        'instruction': debate_instruction1_1,
        'final_decision_instruction': "Sub-task 1_1: Conclude the implications on spectrum continuity and endpoint.",
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'context_desc': ["user query", "thinking of subtask 0_1", "answer of subtask 0_1"],
        'temperature': 0.5
    }
    results1_1, log1_1 = await self.debate(
        subtask_id="subtask_1_1",
        debate_desc=debate_desc1_1,
        n_repeat=self.max_round
    )
    logs.append(log1_1)

    cot_sc_instruction1_2 = (
        "Sub-task 1_2: Based on the outputs from Sub-task 0_1 and Sub-task 1_1, integrate physical principles and decay characteristics "
        "to qualitatively describe how the shape and endpoint of the E particle energy spectrum adjust in the variant decay compared to the original."
    )
    cot_sc_desc1_2 = {
        'instruction': cot_sc_instruction1_2,
        'final_decision_instruction': "Sub-task 1_2: Synthesize and choose the most consistent description of spectral changes.",
        'input': [taskInfo, results0['thinking'], results0['answer'], results1_1['thinking'], results1_1['answer']],
        'temperature': 0.5,
        'context_desc': ["user query", "thinking of subtask 0_1", "answer of subtask 0_1", "thinking of subtask 1_1", "answer of subtask 1_1"]
    }
    results1_2, log1_2 = await self.sc_cot(
        subtask_id="subtask_1_2",
        cot_agent_desc=cot_sc_desc1_2,
        n_repeat=self.max_sc
    )
    logs.append(log1_2)

    debate_instruction2_1 = (
        "Sub-task 2_1: Evaluate the candidate answer choices against the analyzed spectral characteristics "
        "from Sub-tasks 1_1 and 1_2 to select the option that correctly describes the energy spectrum changes of the E particles in the variant decay."
    )
    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'final_decision_instruction': "Sub-task 2_1: Select the best answer choice describing the energy spectrum changes.",
        'input': [taskInfo, results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'context_desc': ["user query", "thinking of subtask 1_1", "answer of subtask 1_1", "thinking of subtask 1_2", "answer of subtask 1_2"],
        'temperature': 0.5
    }
    results2_1, log2_1 = await self.debate(
        subtask_id="subtask_2_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1['thinking'], results2_1['answer'])

    return final_answer, logs
