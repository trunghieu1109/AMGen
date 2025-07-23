async def forward_151(self, taskInfo):
    logs = []
    loop_results_stage_0 = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Extract and summarize the key biological information from the query, including peptide treatment, "
            "shmoo formation, and the experimental approach of chromatin immunoprecipitation followed by mass spectrometry. "
            "Input content are results (both thinking and answer) from: [taskInfo]."
        )
        cot_agent_desc_0_1 = {
            'instruction': cot_instruction_0_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_0_1, log_0_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_0_1
        )
        loop_results_stage_0['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results_stage_0['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Define 'active chromatin' operationally in the context of the experiment by identifying which histone modifications or chromatin-associated factors are targeted by ChIP, "
            "and clarify yeast-specific chromatin biology features relevant to the query, including the absence of canonical metazoan enhancer complexes and the chromatin binding behavior of pre-replication complex proteins during G1 phase. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1, respectively."
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_1']['thinking'] + loop_results_stage_0['stage_0.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results_stage_0['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results_stage_0['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Analyze the roles and relevance of each protein complex option (pre-initiation complex, pre-replication complex, enhancer protein complex, nucleosome histone complex) "
            "in the context of the defined active chromatin and yeast-specific chromatin features. Explicitly evaluate which complexes are constitutively chromatin-bound in G1 and which are species- or stimulus-specific, "
            "to avoid misapplication of metazoan concepts and erroneous assumptions about complex presence. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_1 & stage_0.subtask_2, respectively."
        )
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_1']['thinking'] + loop_results_stage_0['stage_0.subtask_1']['answer'] + loop_results_stage_0['stage_0.subtask_2']['thinking'] + loop_results_stage_0['stage_0.subtask_2']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_0_3
        )
        loop_results_stage_0['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results_stage_0['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        cot_instruction_0_4 = (
            "Sub-task 4: Evaluate which protein complex is least likely to be present or recovered in the active chromatin proteome under the experimental conditions described, "
            "based on the refined analysis. This subtask explicitly incorporates the failure reason of incorrect assumptions about pre-RC absence and misinterpretation of enhancer complexes. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_3, respectively."
        )
        cot_agent_desc_0_4 = {
            'instruction': cot_instruction_0_4,
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_3']['thinking'] + loop_results_stage_0['stage_0.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3']
        }
        results_0_4, log_0_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_0_4
        )
        loop_results_stage_0['stage_0.subtask_4']['thinking'].append(results_0_4['thinking'])
        loop_results_stage_0['stage_0.subtask_4']['answer'].append(results_0_4['answer'])
        logs.append(log_0_4)

        cot_agent_instruction_0_5 = (
            "Sub-task 5: Document the detailed reasoning process and preliminary conclusion about the least represented protein complex in the assay, "
            "ensuring clarity and explicit justification to prevent misinterpretation in later stages. "
            "Input content are results (both thinking and answer) from: stage_0.subtask_4, respectively."
        )
        cot_agent_desc_0_5 = {
            'instruction': cot_agent_instruction_0_5,
            'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_4']['thinking'] + loop_results_stage_0['stage_0.subtask_4']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_0_5, log_0_5 = await self.answer_generate(
            subtask_id='stage_0.subtask_5',
            cot_agent_desc=cot_agent_desc_0_5
        )
        loop_results_stage_0['stage_0.subtask_5']['thinking'].append(results_0_5['thinking'])
        loop_results_stage_0['stage_0.subtask_5']['answer'].append(results_0_5['answer'])
        logs.append(log_0_5)

    aggregate_instruction_1_1 = (
        "Sub-task 1: Combine the summarized biological context, operational definition of active chromatin, yeast-specific chromatin features, "
        "and protein complex analysis from stage_0 to form a consolidated and coherent understanding of the query and experimental context. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively."
    )
    aggregate_desc_1_1 = {
        'instruction': aggregate_instruction_1_1,
        'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_5']['thinking'] + loop_results_stage_0['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_1_1, log_1_1 = await self.aggregate(
        subtask_id='stage_1.subtask_1',
        aggregate_desc=aggregate_desc_1_1
    )
    logs.append(log_1_1)

    cot_instruction_1_2 = (
        "Sub-task 2: Integrate knowledge about chromatin immunoprecipitation and mass spectrometry techniques with the consolidated biological understanding "
        "to assess which protein complexes are expected to co-purify with active chromatin in yeast shmoo cells. "
        "Input content are results (both thinking and answer) from: stage_0.subtask_5, respectively."
    )
    cot_agent_desc_1_2 = {
        'instruction': cot_instruction_1_2,
        'input': [taskInfo] + loop_results_stage_0['stage_0.subtask_5']['thinking'] + loop_results_stage_0['stage_0.subtask_5']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
    }
    results_1_2, log_1_2 = await self.cot(
        subtask_id='stage_1.subtask_2',
        cot_agent_desc=cot_agent_desc_1_2
    )
    logs.append(log_1_2)

    cot_sc_instruction_2_1 = (
        "Sub-task 1: Identify and select the protein complex least represented in the active chromatin proteome based on the consolidated biological and experimental inputs, "
        "ensuring the selection is justified with respect to yeast biology and ChIP-MS assay specifics. "
        "Input content are results (both thinking and answer) from: stage_1.subtask_1 & stage_1.subtask_2, respectively."
    )
    final_decision_instruction_2_1 = (
        "Sub-task 1: Synthesize and choose the most consistent answer for the protein complex least represented in the active chromatin proteome."
    )
    cot_sc_desc_2_1 = {
        'instruction': cot_sc_instruction_2_1,
        'final_decision_instruction': final_decision_instruction_2_1,
        'input': [taskInfo, results_1_1['thinking'], results_1_1['answer'], results_1_2['thinking'], results_1_2['answer']],
        'temperature': 0.5,
        'context_desc': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results_2_1, log_2_1 = await self.sc_cot(
        subtask_id='stage_2.subtask_1',
        cot_agent_desc=cot_sc_desc_2_1,
        n_repeat=self.max_sc
    )
    logs.append(log_2_1)

    review_instruction_3_1 = (
        "Sub-task 1: Evaluate the selected protein complex for correctness and consistency with established biological knowledge, yeast-specific chromatin context, "
        "and the experimental design, explicitly checking for previous reasoning errors such as misinterpretation of pre-RC presence and enhancer complex relevance. "
        "Input content are results (both thinking and answer) from: stage_2.subtask_1, respectively."
    )
    review_desc_3_1 = {
        'instruction': review_instruction_3_1,
        'input': [taskInfo, results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_3_1, log_3_1 = await self.review(
        subtask_id='stage_3.subtask_1',
        review_desc=review_desc_3_1
    )
    logs.append(log_3_1)

    cot_reflect_instruction_4_1 = (
        "Sub-task 1: Refine the final answer by consolidating validation feedback and format it clearly and concisely according to the query requirements, "
        "ensuring the reasoning explicitly addresses and avoids prior failure reasons. "
        "Input content are results (both thinking and answer) from: stage_3.subtask_1 & stage_2.subtask_1, respectively."
    )
    critic_instruction_4_1 = (
        "Please review and provide the limitations of provided solutions of the selected protein complex and suggest improvements if any."
    )
    cot_reflect_desc_4_1 = {
        'instruction': cot_reflect_instruction_4_1,
        'critic_instruction': critic_instruction_4_1,
        'input': [taskInfo, results_3_1['thinking'], results_3_1['answer'], results_2_1['thinking'], results_2_1['answer']],
        'temperature': 0.0,
        'context_desc': ['user query', 'thinking of stage_3.subtask_1', 'answer of stage_3.subtask_1', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results_4_1, log_4_1 = await self.reflexion(
        subtask_id='stage_4.subtask_1',
        reflect_desc=cot_reflect_desc_4_1,
        n_repeat=self.max_round
    )
    logs.append(log_4_1)

    final_answer = await self.make_final_answer(results_4_1['thinking'], results_2_1['answer'])

    return final_answer, logs
