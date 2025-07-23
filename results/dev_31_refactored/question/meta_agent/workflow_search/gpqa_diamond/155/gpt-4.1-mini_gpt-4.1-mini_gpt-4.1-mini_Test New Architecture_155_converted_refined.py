async def forward_155(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            f"Sub-task stage_0.subtask_1: Analyze the stereochemical outcome of Reaction 1: epoxidation of (E)-oct-4-ene with mCPBA followed by aqueous acid treatment, "
            f"including identification of stereocenters formed and preliminary stereoisomer structures. Explicitly consider the stereochemical course and possible diastereomers formed. "
            f"Input content: taskInfo"
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(subtask_id='stage_0.subtask_1', cot_agent_desc=cot_agent_desc_0_1)
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            f"Sub-task stage_0.subtask_2: Assess molecular symmetry and chirality of the products from Reaction 1 to determine if any products are meso (achiral) compounds or enantiomeric pairs. "
            f"This step addresses the previous failure of misclassifying meso compounds as chiral enantiomers, ensuring correct stereoisomer classification. "
            f"Input content: taskInfo, thinking and answer from stage_0.subtask_1 from all previous iterations"
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(subtask_id='stage_0.subtask_2', cot_agent_desc=cot_agent_desc_0_2)
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            f"Sub-task stage_0.subtask_3: Analyze the stereochemical outcome of Reaction 2: epoxidation of (Z)-oct-4-ene with mCPBA followed by aqueous acid treatment, "
            f"including identification of stereocenters and preliminary stereoisomer structures. Use insights from Reaction 1 analysis to inform stereochemical reasoning. "
            f"Input content: taskInfo, thinking and answer from stage_0.subtask_2 from all previous iterations"
        )
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_0_3, log_0_3 = await self.cot(subtask_id='stage_0.subtask_3', cot_agent_desc=cot_agent_desc_0_3)
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_instruction_0_4 = (
            f"Sub-task stage_0.subtask_4: Assess molecular symmetry and chirality of the products from Reaction 2 to determine if any products are meso (achiral) compounds or enantiomeric pairs. "
            f"This prevents repeating the previous error of misclassification and ensures accurate stereoisomer enumeration. "
            f"Input content: taskInfo, thinking and answer from stage_0.subtask_3 from all previous iterations"
        )
        cot_agent_desc_0_4 = {
            'instruction': cot_instruction_0_4,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_0_4, log_0_4 = await self.cot(subtask_id='stage_0.subtask_4', cot_agent_desc=cot_agent_desc_0_4)
        loop_results['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

    cot_instruction_1_1 = (
        f"Sub-task stage_1.subtask_1: Combine the stereochemical products from both reactions, using the symmetry and chirality assessments, "
        f"to determine the total number and types of stereoisomers present in the mixture. Explicitly distinguish between enantiomers, diastereomers, and meso compounds to avoid overcounting, addressing the previous failure. "
        f"Input content: taskInfo, thinking and answer from all iterations of stage_0.subtask_2 and stage_0.subtask_4"
    )
    cot_agent_desc_1_1 = {
        'instruction': cot_instruction_1_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
    }
    results_1_1, log_1_1 = await self.cot(subtask_id='stage_1.subtask_1', cot_agent_desc=cot_agent_desc_1_1)
    logs.append(log_1_1)

    cot_instruction_2_1 = (
        f"Sub-task stage_2.subtask_1: Predict the number of peaks observed in standard (achiral) and chiral HPLC chromatograms based on the correctly enumerated stereoisomeric mixture "
        f"and assuming ideal chromatographic resolution. Use the distinctions between meso, enantiomers, and diastereomers to determine which species are separable under each chromatographic condition. "
        f"Input content: taskInfo, thinking and answer from stage_1.subtask_1"
    )
    cot_agent_desc_2_1 = {
        'instruction': cot_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_2_1, log_2_1 = await self.cot(subtask_id='stage_2.subtask_1', cot_agent_desc=cot_agent_desc_2_1)
    logs.append(log_2_1)

    final_answer = await self.make_final_answer(results_2_1['thinking'], results_2_1['answer'])
    return final_answer, logs
