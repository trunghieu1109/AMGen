async def forward_188(self, taskInfo):
    logs = []
    loop_results = {
        'stage_0.subtask_1': {'thinking': [], 'answer': []},
        'stage_0.subtask_2': {'thinking': [], 'answer': []},
        'stage_0.subtask_3': {'thinking': [], 'answer': []},
        'stage_1.subtask_1': {'thinking': [], 'answer': []},
        'stage_1.subtask_2': {'thinking': [], 'answer': []},
        'stage_2.subtask_1': {'thinking': [], 'answer': []},
        'stage_2.subtask_2': {'thinking': [], 'answer': []},
        'stage_2.subtask_3': {'thinking': [], 'answer': []}
    }

    for iteration in range(2):
        cot_instruction_0_1 = (
            "Sub-task 1: Precisely define and enumerate all possible ways an effective particle or excitation can be 'associated with spontaneously-broken symmetry' (SSB), "
            "including but not limited to: being a Nambu-Goldstone boson, mass generation due to SSB, and existence as a topological defect or soliton whose stability depends on the broken symmetry. "
            "This subtask addresses the previous failure of equating 'association' solely with NG modes and ensures a broad conceptual foundation for subsequent analysis. "
            "Input content: taskInfo"
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
        loop_results['stage_0.subtask_1']['thinking'].append(results_0_1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results_0_1['answer'])
        logs.append(log_0_1)

        cot_instruction_0_2 = (
            "Sub-task 2: Analyze the physical nature and theoretical context of each given particle (Magnon, Skyrmion, Pion, Phonon), focusing on their known relationships to spontaneous symmetry breaking. "
            "Use the broad definition from stage_0.subtask_1 to guide the analysis. This subtask avoids the previous error of excluding topological solitons by considering all forms of association with SSB. "
            "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1, and former iterations of stage_2.subtask_3"
        )
        cot_agent_desc_0_2 = {
            'instruction': cot_instruction_0_2,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_2.subtask_3']['thinking'] + loop_results['stage_2.subtask_3']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of previous stage_2.subtask_3', 'answer of previous stage_2.subtask_3']
        }
        results_0_2, log_0_2 = await self.cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_0_2
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results_0_2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results_0_2['answer'])
        logs.append(log_0_2)

        cot_instruction_0_3 = (
            "Sub-task 3: Explicitly test each particle against every aspect of the association criteria defined in stage_0.subtask_1, determining whether and how each particle is associated with spontaneously-broken symmetry. "
            "This subtask prevents conflation of 'association' with only NG bosons and ensures topological excitations like Skyrmions are correctly classified. It also incorporates feedback about the subtlety of topological defects' dependence on broken symmetry. "
            "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1, stage_0.subtask_2, and former iterations of stage_1.subtask_1"
        )
        cot_agent_desc_0_3 = {
            'instruction': cot_instruction_0_3,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2', 'thinking of previous stage_1.subtask_1', 'answer of previous stage_1.subtask_1']
        }
        results_0_3, log_0_3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_0_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results_0_3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results_0_3['answer'])
        logs.append(log_0_3)

        aggregate_instruction_1_1 = (
            "Sub-task 1: Aggregate and synthesize the detailed definitions and analyses from stage_0 subtasks to form a comprehensive, consolidated view of each particle's association with spontaneously-broken symmetry. "
            "This subtask explicitly incorporates the broader conceptual framework and testing results to avoid the previous narrow interpretation. It also integrates prior iteration results to maintain continuity. "
            "Input content: taskInfo, all thinking and answers from stage_0.subtask_1, stage_0.subtask_2, stage_0.subtask_3, and former iterations of stage_1.subtask_1"
        )
        aggregate_desc_1_1 = {
            'instruction': aggregate_instruction_1_1,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'solutions generated from stage_0 subtasks', 'previous stage_1.subtask_1']
        }
        results_1_1, log_1_1 = await self.aggregate(
            subtask_id='stage_1.subtask_1',
            aggregate_desc=aggregate_desc_1_1
        )
        loop_results['stage_1.subtask_1']['thinking'].append(results_1_1['thinking'])
        loop_results['stage_1.subtask_1']['answer'].append(results_1_1['answer'])
        logs.append(log_1_1)

        cot_instruction_1_2 = (
            "Sub-task 2: Apply clear evaluation criteria derived from the consolidated view to preliminarily classify each particle as associated or not associated with spontaneously-broken symmetry. "
            "This classification must consider all forms of association, including topological dependence, to avoid previous misclassification errors. It also uses prior iteration results for refinement. "
            "Input content: taskInfo, thinking and answer from stage_1.subtask_1, and former iterations of stage_2.subtask_1"
        )
        cot_agent_desc_1_2 = {
            'instruction': cot_instruction_1_2,
            'input': [taskInfo] + loop_results['stage_1.subtask_1']['thinking'] + loop_results['stage_1.subtask_1']['answer'] + loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_1.subtask_1', 'answer of stage_1.subtask_1', 'thinking of previous stage_2.subtask_1', 'answer of previous stage_2.subtask_1']
        }
        results_1_2, log_1_2 = await self.cot(
            subtask_id='stage_1.subtask_2',
            cot_agent_desc=cot_agent_desc_1_2
        )
        loop_results['stage_1.subtask_2']['thinking'].append(results_1_2['thinking'])
        loop_results['stage_1.subtask_2']['answer'].append(results_1_2['answer'])
        logs.append(log_1_2)

        review_instruction_2_1 = (
            "Sub-task 1: Validate the preliminary classifications against rigorous theoretical definitions and known physics principles, explicitly checking subtle conceptual nuances such as the role of topological solitons and their dependence on spontaneously-broken symmetry. "
            "This subtask addresses the previous failure to recognize Skyrmions' association with SSB and incorporates prior iteration results for iterative improvement. "
            "Input content: taskInfo, thinking and answer from stage_1.subtask_2, and former iterations of stage_2.subtask_1"
        )
        review_desc_2_1 = {
            'instruction': review_instruction_2_1,
            'input': [taskInfo] + loop_results['stage_1.subtask_2']['thinking'] + loop_results['stage_1.subtask_2']['answer'] + loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer'],
            'temperature': 0.0,
            'context': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2', 'thinking of previous stage_2.subtask_1', 'answer of previous stage_2.subtask_1']
        }
        results_2_1, log_2_1 = await self.review(
            subtask_id='stage_2.subtask_1',
            review_desc=review_desc_2_1
        )
        loop_results['stage_2.subtask_1']['thinking'].append(results_2_1['thinking'])
        loop_results['stage_2.subtask_1']['answer'].append(results_2_1['answer'])
        logs.append(log_2_1)

        debate_instruction_2_2 = (
            "Sub-task 2: Select the particle(s) that do not fit the criteria of being associated with spontaneously-broken symmetry based on the validated classifications. "
            "This selection must be justified with reference to the broad association framework and validation results to avoid premature exclusion of particles like Skyrmions. "
            "Input content: taskInfo, thinking and answer from stage_2.subtask_1"
        )
        debate_desc_2_2 = {
            'instruction': debate_instruction_2_2,
            'final_decision_instruction': "Sub-task 2: Select particle(s) not associated with spontaneously-broken symmetry based on validation results.",
            'input': [taskInfo] + loop_results['stage_2.subtask_1']['thinking'] + loop_results['stage_2.subtask_1']['answer'],
            'context_desc': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1'],
            'temperature': 0.5
        }
        results_2_2, log_2_2 = await self.debate(
            subtask_id='stage_2.subtask_2',
            debate_desc=debate_desc_2_2,
            n_repeat=self.max_round
        )
        loop_results['stage_2.subtask_2']['thinking'].append(results_2_2['thinking'])
        loop_results['stage_2.subtask_2']['answer'].append(results_2_2['answer'])
        logs.append(log_2_2)

        cot_sc_instruction_2_3 = (
            "Sub-task 3: Assess the validity and consistency of the selection to ensure the correct particle is identified as not associated with spontaneously-broken symmetry. "
            "This subtask uses a stochastic chain-of-thought approach to robustly confirm the final choice and incorporates prior iteration results to prevent repeated errors. "
            "Input content: taskInfo, thinking and answer from stage_2.subtask_2, and former iterations of stage_2.subtask_3"
        )
        final_decision_instruction_2_3 = "Sub-task 3: Synthesize and choose the most consistent answer for the particle not associated with spontaneously-broken symmetry."
        cot_sc_desc_2_3 = {
            'instruction': cot_sc_instruction_2_3,
            'final_decision_instruction': final_decision_instruction_2_3,
            'input': [taskInfo] + loop_results['stage_2.subtask_2']['thinking'] + loop_results['stage_2.subtask_2']['answer'] + loop_results['stage_2.subtask_3']['thinking'] + loop_results['stage_2.subtask_3']['answer'],
            'temperature': 0.5,
            'context_desc': ['user query', 'thinking of stage_2.subtask_2', 'answer of stage_2.subtask_2', 'thinking of previous stage_2.subtask_3', 'answer of previous stage_2.subtask_3']
        }
        results_2_3, log_2_3 = await self.sc_cot(
            subtask_id='stage_2.subtask_3',
            cot_agent_desc=cot_sc_desc_2_3,
            n_repeat=self.max_sc
        )
        loop_results['stage_2.subtask_3']['thinking'].append(results_2_3['thinking'])
        loop_results['stage_2.subtask_3']['answer'].append(results_2_3['answer'])
        logs.append(log_2_3)

    formatter_instruction_3_1 = (
        "Sub-task 1: Consolidate the validated selection into a clear, concise final answer specifying which particle is not associated with spontaneously-broken symmetry. "
        "The output must follow the required format and reflect the refined understanding developed through all prior stages. "
        "Input content: taskInfo, thinking and answer from stage_2.subtask_3"
    )
    formatter_desc_3_1 = {
        'instruction': formatter_instruction_3_1,
        'input': [taskInfo] + loop_results['stage_2.subtask_3']['thinking'] + loop_results['stage_2.subtask_3']['answer'],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_3', 'answer of stage_2.subtask_3'],
        'format': 'short and concise, without explanation'
    }
    results_3_1, log_3_1 = await self.specific_format(
        subtask_id='stage_3.subtask_1',
        formatter_desc=formatter_desc_3_1
    )
    logs.append(log_3_1)

    final_answer = await self.make_final_answer(results_3_1['thinking'], results_3_1['answer'])
    return final_answer, logs
