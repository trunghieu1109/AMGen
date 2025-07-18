async def forward_196(self, taskInfo):
    logs = []

    cot_sc_instruction0_1 = (
        "Sub-task 1: Extract and explicitly assign each IR and 1H NMR spectral feature to specific functional groups and structural elements of compound X, "
        "with clear justification. Confirm presence of carboxylic acid group (broad 3400–2500 cm⁻¹ O–H stretch, 1720 cm⁻¹ C=O stretch, 10.5 ppm broad singlet proton), aromatic substitution pattern (two doublets at 8.0 and 7.2 ppm integrating to 2H each indicating para-substitution), "
        "and nature of alkyl side chain (multiplets and splitting patterns consistent with sec-butyl). Block progress until this summary is complete and validated to avoid assumptions about other functional groups."
    )

    cot_sc_desc0_1 = {
        'instruction': cot_sc_instruction0_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }

    results0_1, log0_1 = await self.sc_cot(
        subtask_id="stage_0.subtask_1",
        cot_agent_desc=cot_sc_desc0_1,
        n_repeat=self.max_sc
    )
    logs.append(log0_1)

    cot_sc_instruction1_1 = (
        "Sub-task 1: Based strictly on the confirmed functional groups and structural features from Stage 0, analyze the reaction conditions (red phosphorus and HI) "
        "to predict the chemical transformation of compound X. Explicitly consider the known reaction pathway of P/HI with carboxylic acids (decarboxylation/reduction to hydrocarbons or alkyl halides) rather than generic reductions. "
        "Provide a detailed mechanistic rationale linking the spectral data and reaction conditions to the expected product structure."
    )

    cot_sc_desc1_1 = {
        'instruction': cot_sc_instruction1_1,
        'input': [taskInfo, results0_1['thinking'], results0_1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1"]
    }

    results1_1, log1_1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1_1,
        n_repeat=self.max_sc
    )
    logs.append(log1_1)

    cot_reflect_instruction1_2 = (
        "Sub-task 2: Based on the confirmed functional groups and predicted product from subtask_1, map the predicted product structure to the given multiple-choice options. "
        "Compare key structural features such as aromatic substitution pattern, presence or absence of carboxylic acid, and alkyl substituent identity (sec-butyl vs isobutyl, methyl vs ethyl) to select the most consistent final product. "
        "Filter valid scenarios that meet the conditions stated in the query."
    )

    cot_reflect_desc1_2 = {
        'instruction': cot_reflect_instruction1_2,
        'input': [taskInfo, results0_1['thinking'], results0_1['answer'], results1_1['thinking'], results1_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of stage_0.subtask_1", "answer of stage_0.subtask_1", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }

    results1_2, log1_2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc1_2,
        n_repeat=self.max_round
    )
    logs.append(log1_2)

    debate_instruction2_1 = (
        "Sub-task 1: Evaluate and prioritize the candidate products by critically comparing their structural features with the predicted product from Stage 1, "
        "using all spectral and mechanistic evidence. Engage in a reasoned debate to resolve any ambiguities or close matches, ensuring the final product identification is robust and justified. "
        "Consolidate all prior reasoning and prevent premature or unsupported conclusions."
    )

    debate_desc2_1 = {
        'instruction': debate_instruction2_1,
        'context': ["user query", results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'input': [taskInfo, results1_1['thinking'], results1_1['answer'], results1_2['thinking'], results1_2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }

    results2_1, log2_1 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc2_1,
        n_repeat=self.max_round
    )
    logs.append(log2_1)

    final_answer = await self.make_final_answer(results2_1['thinking'], results2_1['answer'])

    return final_answer, logs
