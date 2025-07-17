async def forward_152(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize all given chemical information, including reactants, reagents, reaction conditions, "
        "and definitions related to Michael addition reactions from the provided query. Ensure a clear understanding of the problem context "
        "and the chemical species involved to support subsequent mechanistic reasoning."
    )
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Derive the expected major products and intermediates of the three Michael addition reactions by applying detailed mechanistic reasoning. "
        "Analyze regioselectivity, resonance stabilization, and keto–enol tautomerization equilibria under the given reaction and workup conditions, especially for 1,3-diketones in reaction C. "
        "Avoid misassigning tautomeric forms by critically evaluating which tautomer predominates."
    )
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.debate(
        subtask_id="stage_1.subtask_2",
        debate_desc=debate_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    reflexion_instruction3 = (
        "Sub-task 3: Perform a self-consistency and reflective cross-check of the derived products focusing on tautomeric equilibria, resonance stabilization, and chemical plausibility. "
        "Explicitly challenge assumptions made in Sub-task 2, particularly regarding tautomeric forms and product stability, to prevent propagation of errors."
    )
    critic_instruction3 = (
        "Please review the valid scenarios filtering and provide its limitations."
    )
    cot_reflect_desc3 = {
        'instruction': reflexion_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': [
            "user query",
            "thinking of stage_1.subtask_1",
            "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2",
            "answer of stage_1.subtask_2"
        ]
    }
    results3, log3 = await self.reflexion(
        subtask_id="stage_1.subtask_3",
        reflect_desc=cot_reflect_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Stage 2 Sub-task 1: Evaluate and prioritize the four multiple-choice options by comparing their proposed products with the rigorously derived and cross-checked products from stage_1. "
        "Incorporate a debate-style critical assessment to verify chemical plausibility, tautomeric correctness, and consistency with mechanistic reasoning. "
        "Explicitly re-examine any prior assumptions and ensure the final choice aligns with the most chemically accurate interpretation of the reactions."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': [
            "user query",
            "thinking of stage_1.subtask_1",
            "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_2",
            "answer of stage_1.subtask_2",
            "thinking of stage_1.subtask_3",
            "answer of stage_1.subtask_3"
        ],
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
