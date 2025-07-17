async def forward_196(self, taskInfo):
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and summarize the defining spectral features and structural information of compound X from the given IR and 1H NMR data. "
        "Ensure accurate identification of functional groups and substitution patterns without presuming the reaction outcome. "
        "This subtask must avoid bias toward any reaction mechanism and focus solely on reliable spectral interpretation."
    )
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1, log1 = await self.cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    logs.append(log1)

    debate_instruction2 = (
        "Sub-task 2: Enumerate and critically assess all plausible chemical transformation pathways of compound X upon reaction with red phosphorus and HI, "
        "explicitly including decarboxylation/reduction and halogenation mechanisms. Justify each pathway by referencing well-established organic chemistry principles and textbook examples. "
        "This subtask must avoid premature commitment to a single mechanism and instead weigh competing mechanisms to prevent the previous error of misassigning the reaction outcome."
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
        "Sub-task 3: Conduct a critical reflection or mini-debate to select the most chemically plausible reaction mechanism from the enumerated pathways in subtask_2. "
        "This step must explicitly verify the chosen mechanism against literature precedent for aromatic carboxylic acids treated with red phosphorus and HI, "
        "ensuring the reaction outcome is correctly identified as decarboxylation/reduction rather than halogenation. "
        "This prevents propagation of conceptual errors into product identification."
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
        "Stage 2 Sub-task 1: Evaluate and prioritize the given candidate final products by comparing their structures and substituents against the correctly inferred product structure from the combined spectral analysis (stage_1.subtask_1) "
        "and validated chemical transformation mechanism (stage_1.subtask_3). This subtask must ensure that the final product matches the expected decarboxylated/reduced hydrocarbon derivative, explicitly avoiding selection of the starting acid or incorrect halogenated products."
    )
    debate_desc4 = {
        'instruction': debate_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': [
            "user query",
            "thinking of stage_1.subtask_1",
            "answer of stage_1.subtask_1",
            "thinking of stage_1.subtask_3",
            "answer of stage_1.subtask_3"
        ]
    }
    results4, log4 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
