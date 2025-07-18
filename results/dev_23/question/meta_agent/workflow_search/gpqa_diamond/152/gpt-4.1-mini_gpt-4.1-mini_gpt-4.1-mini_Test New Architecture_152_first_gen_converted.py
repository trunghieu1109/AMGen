async def forward_152(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given chemical information, including reactants, reagents, reaction conditions, "
        "and definitions related to Michael addition reactions from the provided query."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Based on the output from Sub-task 1, classify and analyze the chemical species involved (nucleophiles, electrophiles, intermediates) "
        "and their roles in the given reactions, including the identification of compound C candidates."
    )
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_0.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    cot_reflect_instruction3 = (
        "Sub-task 3: Based on the outputs from Sub-task 1 and Sub-task 2, derive the major final products of each Michael addition reaction (A, B, and C) "
        "by applying mechanistic reasoning, considering resonance stabilization, tautomerism, and reaction conditions."
    )
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': [
            "user query",
            "thinking of stage_0.subtask_1",
            "answer of stage_0.subtask_1",
            "thinking of stage_0.subtask_2",
            "answer of stage_0.subtask_2"
        ]
    }
    results3, log3 = await self.reflexion(
        subtask_id="stage_1.subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Evaluate the four multiple-choice options against the derived products to identify which option correctly matches "
        "the reactants and major final products for reactions A, B, and C."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': [
            "user query",
            "thinking of stage_1.subtask_3",
            "answer of stage_1.subtask_3"
        ],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_2.subtask_4",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
