async def forward_155(self, taskInfo):
    logs = []
    loop_results_stage_0 = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_agent_desc_1 = {
            'instruction': (
                "Sub-task 1: Analyze the stereochemical outcome of Reaction 1: epoxidation of (E)-oct-4-ene with mCPBA and aqueous acid treatment, "
                "identifying the stereoisomers formed. Input content: user query."
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context_desc': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results_stage_0['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results_stage_0['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_agent_desc_2 = {
            'instruction': (
                "Sub-task 2: Analyze the stereochemical outcome of Reaction 2: epoxidation of (Z)-oct-4-ene with mCPBA and aqueous acid treatment, "
                "identifying the stereoisomers formed. Input content: user query, thinking and answer from all previous iterations of stage_0.subtask_1."
            ),
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_1']['thinking'] + loop_results_stage_0['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results_stage_0['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results_stage_0['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_agent_desc_3 = {
            'instruction': (
                "Sub-task 3: Determine the stereochemical relationship between the products of Reaction 1 and Reaction 2, "
                "classifying them as enantiomers or diastereomers. Input content: user query, thinking and answer from all previous iterations of stage_0.subtask_1 and stage_0.subtask_2."
            ),
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_1']['thinking'] + loop_results_stage_0['stage_0.subtask_1']['answer'] + loop_results_stage_0['stage_0.subtask_2']['thinking'] + loop_results_stage_0['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results_stage_0['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results_stage_0['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_agent_desc_4 = {
            'instruction': (
                "Sub-task 4: Predict the total number of stereoisomers present in the combined product mixture from both reactions. "
                "Input content: user query, thinking and answer from all previous iterations of stage_0.subtask_3."
            ),
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_3']['thinking'] + loop_results_stage_0['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_0_4, log_0_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results_stage_0['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results_stage_0['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

        cot_agent_desc_5 = {
            'instruction': (
                "Sub-task 5: Summarize the stereochemical composition of the combined product mixture, including the nature of stereoisomers relevant for chromatographic separation. "
                "Input content: user query, thinking and answer from all previous iterations of stage_0.subtask_4."
            ),
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_4']['thinking'] + loop_results_stage_0['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context_desc': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_0_5, log_0_5 = await self.answer_generate(
            subtask_id='stage_0.subtask_5',
            cot_agent_desc=cot_agent_desc_5
        )
        loop_results_stage_0['stage_0.subtask_5']['thinking'].append(results_0_5['thinking'])
        loop_results_stage_0['stage_0.subtask_5']['answer'].append(results_0_5['answer'])
        logs.append(log_0_5)

    aggregate_desc_1 = {
        'instruction': (
            "Sub-task 1: Combine the stereochemical analyses from stage_0 to form a comprehensive description of the product mixture's stereoisomeric composition. "
            "Input content: user query, thinking and answer from all iterations of stage_0.subtask_5."
        ),
        'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_5']['thinking'] + loop_results_stage_0['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context_desc': ['user query', 'solutions generated from stage_0.subtask_5']
    }
    results_1_1, log_1_1 = await self.aggregate(
        subtask_id='stage_1.subtask_1',
        aggregate_desc=aggregate_desc_1
    )
    logs.append(log_1_1)

    cot_agent_desc_2 = {
        'instruction': (
            "Sub-task 2: Integrate knowledge of chromatographic principles (achiral and chiral HPLC) with the stereochemical composition to predict separation behavior. "
            "Input content: user query, thinking and answer from stage_1.subtask_1."
        ),
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1']
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id='stage_1.subtask_2',
        cot_agent_desc=cot_agent_desc_2
    )
    logs.append(log_1_2)

    cot_agent_desc_3 = {
        'instruction': (
            "Sub-task 3: Identify which stereoisomers are expected to be resolved by achiral HPLC and which require chiral HPLC for separation. "
            "Input content: user query, thinking and answer from stage_1.subtask_2."
        ),
        'input': [taskInfo, results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_1_3, log_1_3 = await self.cot(
        subtask_id='stage_1.subtask_3',
        cot_agent_desc=cot_agent_desc_3
    )
    logs.append(log_1_3)

    cot_sc_desc_1 = {
        'instruction': (
            "Sub-task 1: Select the number of distinct peaks expected in the standard (achiral) reverse-phase HPLC chromatogram based on stereoisomeric differences. "
            "Input content: user query, thinking and answer from stage_1.subtask_3."
        ),
        'final_decision_instruction': (
            "Sub-task 1: Synthesize and choose the most consistent answer for the number of peaks in achiral HPLC chromatogram."
        ),
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_1, log_2_1 = await self.sc_cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_sc_desc_1,
        n_repeat=self.max_sc
    )
    logs.append(log_2_1)

    cot_sc_desc_2 = {
        'instruction': (
            "Sub-task 2: Select the number of distinct peaks expected in the chiral HPLC chromatogram based on enantiomeric and diastereomeric resolution. "
            "Input content: user query, thinking and answer from stage_1.subtask_3."
        ),
        'final_decision_instruction': (
            "Sub-task 2: Synthesize and choose the most consistent answer for the number of peaks in chiral HPLC chromatogram."
        ),
        'input': [taskInfo, results_1_3['thinking'], results_1_3['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_3', 'answer of stage_1.subtask_3']
    }
    results_2_2, log_2_2 = await self.sc_cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_sc_desc_2,
        n_repeat=self.max_sc
    )
    logs.append(log_2_2)

    cot_agent_desc_3 = {
        'instruction': (
            "Sub-task 3: Compare and contrast the expected peak counts from both chromatographic methods to identify differences in resolution capabilities. "
            "Input content: user query, thinking and answer from stage_2.subtask_1 and stage_2.subtask_2."
        ),
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer'], results_2_2['thinking'], results_2_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2']
    }
    results_2_3, log_2_3 = await self.cot(
        subtask_id='stage_2.subtask_3',
        cot_agent_desc=cot_agent_desc_3
    )
    logs.append(log_2_3)

    review_desc_1 = {
        'instruction': (
            "Sub-task 1: Evaluate the consistency of the predicted chromatographic peak counts with known stereochemical and chromatographic principles. "
            "Input content: user query, thinking and answer from stage_2.subtask_3."
        ),
        'input': [taskInfo, results_2_3['thinking'], results_2_3['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_2.subtask_3', 'answer of stage_2.subtask_3']
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_1
    )
    logs.append(log_3_1)

    cot_agent_desc_2 = {
        'instruction': (
            "Sub-task 2: Assess whether the predicted chromatograms align with the theoretical maximum chromatographic resolution assumption. "
            "Input content: user query, thinking and answer from stage_3.subtask_1."
        ),
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1']
    }
    results_3_2, log_3_2 = await self.cot(
        subtask_id='stage_3.subtask_2',
        cot_agent_desc=cot_agent_desc_2
    )
    logs.append(log_3_2)

    cot_reflect_desc_1 = {
        'instruction': (
            "Sub-task 1: Refine the chromatographic predictions into a clear, concise summary suitable for final answer selection. "
            "Input content: user query, thinking and answer from stage_3.subtask_2."
        ),
        'critic_instruction': (
            "Please review and provide limitations or improvements for the refined chromatographic predictions."
        ),
        'input': [taskInfo, results_3_2['thinking'], results_3_2['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_2', 'answer of stage_3.subtask_2']
    }
    results_4_1, log_4_1 = await self.reflexion(
        subtask_id='stage_4.subtask_1',
        reflect_desc=cot_reflect_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log_4_1)

    formatter_desc_2 = {
        'instruction': (
            "Sub-task 2: Format the final answer by matching the predicted chromatographic observations to one of the provided multiple-choice options. "
            "Input content: user query, thinking and answer from stage_4.subtask_1."
        ),
        'input': [taskInfo, results_4_1['thinking'], results_4_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_4.subtask_1', 'answer of stage_4.subtask_1'],
        'format': 'short and concise, without explanation'
    }
    results_4_2, log_4_2 = await self.specific_format(
        subtask_id='stage_4.subtask_2',
        formatter_desc=formatter_desc_2
    )
    logs.append(log_4_2)

    final_answer = await self.make_final_answer(results_4_2['thinking'], results_4_2['answer'])
    return final_answer, logs
