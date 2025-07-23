async def forward_157(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_0.subtask_4': {'thinking': [], 'answer': []},
        'stage_0.subtask_5': {'thinking': [], 'answer': []},
        'stage_0.subtask_6': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_1 = (
            "Sub-task 1: Extract and summarize all relevant information from the query, including mutation locations, mutation types, protein domains, and functional consequences. "
            "Explicitly note the recessive vs dominant-negative nature of mutations and their domain context to avoid misinterpretation in later steps. "
            "Input content: [taskInfo]"
        )
        cot_agent_desc_1 = {
            'instruction': cot_instruction_1,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results_1, log_1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_1['answer'])
        logs.append(log_1)

        cot_instruction_2 = (
            "Sub-task 2: Analyze the relationships between phosphorylation, dimerization, nuclear translocation, and transcriptional activation, "
            "integrating the summarized information from stage_0.subtask_1 and previous iterations of stage_0.subtask_5. "
            "Emphasize the biological sequence and functional dependencies to clarify the mechanism. "
            "Input content: [taskInfo, all previous thinking and answers from stage_0.subtask_1 and stage_0.subtask_5]"
        )
        cot_agent_desc_2 = {
            'instruction': cot_instruction_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results_2, log_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_2['answer'])
        logs.append(log_2)

        cot_instruction_3 = (
            "Sub-task 3: Map the functional roles of the transactivation and dimerization domains and interpret how mutations X and Y affect these functions based on their domain locations and mutation types. "
            "Incorporate explicit distinctions between recessive loss-of-function and dominant-negative effects to prevent conflation. "
            "Input content: [taskInfo, all previous thinking and answers from stage_0.subtask_2 and stage_0.subtask_5]"
        )
        cot_agent_desc_3 = {
            'instruction': cot_instruction_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results_3, log_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_3['answer'])
        logs.append(log_3)

        cot_instruction_4 = (
            "Sub-task 4: Interpret the molecular mechanism of the dominant-negative mutation Y in the dimerization domain, considering typical dominant-negative effects on protein complexes. "
            "Highlight that dominant-negative mutations often form nonfunctional or aggregated complexes with wild-type subunits rather than simply abolishing dimerization. "
            "Input content: [taskInfo, all previous thinking and answers from stage_0.subtask_3 and stage_0.subtask_5]"
        )
        cot_agent_desc_4 = {
            'instruction': cot_instruction_4,
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results_4, log_4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results_4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results_4['answer'])
        logs.append(log_4)

        cot_instruction_5 = (
            "Sub-task 5: Perform a detailed semantic parsing and verification of each provided molecular phenotype option phrase by phrase. "
            "Explicitly identify and flag any contradictory or ambiguous phrases (e.g., 'loss of protein dimerization and wild-type phenotype') to prevent acceptance of biologically implausible options. "
            "This subtask addresses the previous failure to critically parse answer choices. "
            "Input content: [taskInfo, thinking and answer from stage_0.subtask_4]"
        )
        cot_agent_desc_5 = {
            'instruction': cot_instruction_5,
            'input': [taskInfo, results_4['thinking'], results_4['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results_5, log_5 = await self.cot(
            subtask_id='stage_0.subtask_5',
            cot_agent_desc=cot_agent_desc_5
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results_5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results_5['answer'])
        logs.append(log_5)

        cot_sc_instruction_6 = (
            "Sub-task 6: Evaluate each molecular phenotype option using a structured critique (SC-CoT) approach by listing pros and cons of each choice against the known dominant-negative mutation mechanism and the semantic parsing results from stage_0.subtask_5. "
            "Avoid premature consensus and ensure thorough critical analysis to prevent misinterpretation of options. "
            "Input content: [taskInfo, thinking and answer from stage_0.subtask_5]"
        )
        final_decision_instruction_6 = (
            "Sub-task 6: Evaluate each molecular phenotype option using a structured critique (SC-CoT) approach by listing pros and cons of each choice against the known dominant-negative mutation mechanism and the semantic parsing results from stage_0.subtask_5. "
            "Avoid premature consensus and ensure thorough critical analysis to prevent misinterpretation of options. "
            "Input content: [taskInfo, thinking and answer from stage_0.subtask_5]"
        )
        cot_sc_desc_6 = {
            'instruction': cot_sc_instruction_6,
            'final_decision_instruction': final_decision_instruction_6,
            'input': [taskInfo, results_5['thinking'], results_5['answer']],
            'temperature': 0.5,
            'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results_6, log_6 = await self.sc_cot(
            subtask_id='stage_0.subtask_6',
            cot_agent_desc=cot_sc_desc_6,
            n_repeat=self.max_sc
        )
        loop_results['stage_0.subtask_6']['thinking'].append(results_6['thinking'])
        loop_results['stage_0.subtask_6']['answer'].append(results_6['answer'])
        logs.append(log_6)

    debate_instruction_1 = (
        "Sub-task 1: Select the best molecular phenotype candidate from the evaluated options based on the refined semantic parsing and structured critique outputs from stage_0.subtask_6. "
        "Explicitly reject options with contradictory or ambiguous wording unless a clear, biologically plausible reinterpretation is justified. "
        "This subtask incorporates feedback to avoid acceptance of contradictory options and ensures a robust final decision. "
        "Input content: [taskInfo, thinking and answer from all iterations of stage_0.subtask_6]"
    )
    final_decision_instruction_1 = (
        "Sub-task 1: Select the best molecular phenotype candidate from the evaluated options based on the refined semantic parsing and structured critique outputs from stage_0.subtask_6. "
        "Explicitly reject options with contradictory or ambiguous wording unless a clear, biologically plausible reinterpretation is justified. "
        "This subtask incorporates feedback to avoid acceptance of contradictory options and ensures a robust final decision. "
        "Input content: [taskInfo, thinking and answer from all iterations of stage_0.subtask_6]"
    )
    debate_desc_1 = {
        'instruction': debate_instruction_1,
        'final_decision_instruction': final_decision_instruction_1,
        'input': [taskInfo] + loop_results['stage_0.subtask_6']['thinking'] + loop_results['stage_0.subtask_6']['answer'],
        'context': ['user query', 'thinking of stage_0.subtask_6', 'answer of stage_0.subtask_6'],
        'temperature': 0.5
    }
    results_final, log_final = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=debate_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log_final)

    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'])
    return final_answer, logs
