async def forward_164(self, taskInfo):
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
        cot_agent_desc_1 = {
            'instruction': (
                "Sub-task 1: Extract and summarize all relevant information from the query, including catalyst types, polymerization conditions, "
                "and the four statements to be evaluated. Ensure clarity on the context of homogeneous organometallic catalysts and the goal of introducing regular branches using only ethylene. "
                "Input content: taskInfo"
            ),
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ['user query']
        }
        results1, log1 = await self.cot(
            subtask_id='stage_0.subtask_1',
            cot_agent_desc=cot_agent_desc_1
        )
        loop_results['stage_0.subtask_1']['thinking'].append(results1['thinking'])
        loop_results['stage_0.subtask_1']['answer'].append(results1['answer'])
        logs.append(log1)

        cot_agent_desc_2 = {
            'instruction': (
                "Sub-task 2: Analyze the chemical feasibility and mechanistic relationships between catalyst systems, activators, "
                "and polymer branching mechanisms based on the extracted information. Focus on clarifying the 'essential additional reaction step' and compatibility of activators, avoiding assumptions about industrial implementation. "
                "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1"
            ),
            'final_decision_instruction': (
                "Sub-task 2: Synthesize and choose the most consistent answer for the chemical feasibility and mechanistic relationships analysis."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'temperature': 0.5,
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results2, log2 = await self.sc_cot(
            subtask_id='stage_0.subtask_2',
            cot_agent_desc=cot_agent_desc_2,
            n_repeat=self.max_sc
        )
        loop_results['stage_0.subtask_2']['thinking'].append(results2['thinking'])
        loop_results['stage_0.subtask_2']['answer'].append(results2['answer'])
        logs.append(log2)

        cot_agent_desc_3 = {
            'instruction': (
                "Sub-task 3: Perform a dedicated fact-check on the industrial implementation status of homogeneous dual catalyst systems for producing branched polyethylene from ethylene alone in the US. "
                "Explicitly distinguish homogeneous organometallic systems from heterogeneous Phillips/Ziegler–Natta technologies to avoid conflation. "
                "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_1"
            ),
            'temperature': 0.0,
            'input': [taskInfo] + loop_results['stage_0.subtask_1']['thinking'] + loop_results['stage_0.subtask_1']['answer'],
            'context': ['user query', 'thinking of stage_0.subtask_1', 'answer of stage_0.subtask_1']
        }
        results3, log3 = await self.cot(
            subtask_id='stage_0.subtask_3',
            cot_agent_desc=cot_agent_desc_3
        )
        loop_results['stage_0.subtask_3']['thinking'].append(results3['thinking'])
        loop_results['stage_0.subtask_3']['answer'].append(results3['answer'])
        logs.append(log3)

        cot_agent_desc_4 = {
            'instruction': (
                "Sub-task 4: Verify the mechanistic and chemical correctness of ambiguous or critical statements, especially Statement 2 regarding aluminum-based activators. "
                "Cross-check with authoritative polymer chemistry knowledge and literature to clarify activator compatibility and the nature of the additional reaction step. "
                "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_2"
            ),
            'temperature': 0.0,
            'input': [taskInfo] + loop_results['stage_0.subtask_2']['thinking'] + loop_results['stage_0.subtask_2']['answer'],
            'context': ['user query', 'thinking of stage_0.subtask_2', 'answer of stage_0.subtask_2']
        }
        results4, log4 = await self.cot(
            subtask_id='stage_0.subtask_4',
            cot_agent_desc=cot_agent_desc_4
        )
        loop_results['stage_0.subtask_4']['thinking'].append(results4['thinking'])
        loop_results['stage_0.subtask_4']['answer'].append(results4['answer'])
        logs.append(log4)

        cot_agent_desc_5 = {
            'instruction': (
                "Sub-task 5: Integrate industrial and economic considerations, such as catalyst cost and scalability, into the analysis of the statements, "
                "but only after mechanistic and industrial fact-checks are complete. Explicitly question whether economic factors affect the correctness of statements rather than their relevance. "
                "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_3 and stage_0.subtask_4"
            ),
            'final_decision_instruction': (
                "Sub-task 5: Synthesize and choose the most consistent answer for the integration of industrial and economic considerations."
            ),
            'input': [taskInfo] + loop_results['stage_0.subtask_3']['thinking'] + loop_results['stage_0.subtask_3']['answer'] + loop_results['stage_0.subtask_4']['thinking'] + loop_results['stage_0.subtask_4']['answer'],
            'temperature': 0.5,
            'context': ['user query', 'thinking of stage_0.subtask_3', 'answer of stage_0.subtask_3', 'thinking of stage_0.subtask_4', 'answer of stage_0.subtask_4']
        }
        results5, log5 = await self.debate(
            subtask_id='stage_0.subtask_5',
            debate_desc=cot_agent_desc_5,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_5']['thinking'].append(results5['thinking'])
        loop_results['stage_0.subtask_5']['answer'].append(results5['answer'])
        logs.append(log5)

        cot_reflect_desc_6 = {
            'instruction': (
                "Sub-task 6: Synthesize all previous subtasks' outputs to form a coherent, factually accurate assessment framework for evaluating the four statements individually. "
                "Emphasize factual correctness over industrial relevance or comprehensiveness, addressing the previous error of conflating these aspects. "
                "Input content: taskInfo, all previous thinking and answers from stage_0.subtask_5"
            ),
            'critic_instruction': (
                "Please review and provide the limitations of provided solutions of the assessment framework synthesis."
            ),
            'temperature': 0.0,
            'input': [taskInfo] + loop_results['stage_0.subtask_5']['thinking'] + loop_results['stage_0.subtask_5']['answer'],
            'context': ['user query', 'thinking of stage_0.subtask_5', 'answer of stage_0.subtask_5']
        }
        results6, log6 = await self.reflexion(
            subtask_id='stage_0.subtask_6',
            reflect_desc=cot_reflect_desc_6,
            n_repeat=self.max_round
        )
        loop_results['stage_0.subtask_6']['thinking'].append(results6['thinking'])
        loop_results['stage_0.subtask_6']['answer'].append(results6['answer'])
        logs.append(log6)

    cot_debate_desc_1 = {
        'instruction': (
            "Sub-task 1: Evaluate each of the four statements against the refined assessment framework developed in stage_0.subtask_6 to determine their individual factual correctness. "
            "Explicitly avoid prioritizing industrial implementation or comprehensiveness over correctness, as per previous feedback. "
            "Input content: taskInfo, all thinking and answers from stage_0.subtask_6"
        ),
        'final_decision_instruction': (
            "Sub-task 1: Synthesize and choose the most consistent answer for the evaluation of the four statements."
        ),
        'input': [taskInfo] + loop_results['stage_0.subtask_6']['thinking'] + loop_results['stage_0.subtask_6']['answer'],
        'context': ['user query', 'thinking of stage_0.subtask_6', 'answer of stage_0.subtask_6'],
        'temperature': 0.5
    }
    results7, log7 = await self.debate(
        subtask_id='stage_1.subtask_1',
        debate_desc=cot_debate_desc_1,
        n_repeat=self.max_round
    )
    logs.append(log7)

    aggregate_desc_2 = {
        'instruction': (
            "Sub-task 2: Select the single statement that is factually correct regarding the formation of a polymer with regular branches using only ethylene and a dual catalyst system. "
            "Justify the choice based strictly on chemical, mechanistic, and industrial facts without bias towards economic or implementation scale. "
            "Input content: taskInfo, thinking and answer from stage_1.subtask_1"
        ),
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from stage_1.subtask_1']
    }
    results8, log8 = await self.aggregate(
        subtask_id='stage_1.subtask_2',
        aggregate_desc=aggregate_desc_2
    )
    logs.append(log8)

    review_desc_1 = {
        'instruction': (
            "Sub-task 1: Critically review the selected statement for consistency, correctness, and alignment with known polymer chemistry principles and industrial practices. "
            "Ensure no previous reasoning errors (such as conflation of heterogeneous and homogeneous systems or misinterpretation of question intent) are present. "
            "Input content: taskInfo, thinking and answer from stage_1.subtask_2"
        ),
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_1.subtask_2', 'answer of stage_1.subtask_2']
    }
    results9, log9 = await self.review(
        subtask_id='stage_2.subtask_1',
        review_desc=review_desc_1
    )
    logs.append(log9)

    cot_instruction_2 = {
        'instruction': (
            "Sub-task 2: Provide a final assessment confirming or rejecting the selected statement as the correct answer to the query, explicitly stating the rationale and addressing any potential ambiguities or previous failure reasons. "
            "Input content: taskInfo, thinking and answer from stage_2.subtask_1"
        ),
        'input': [taskInfo, results9['thinking'], results9['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of stage_2.subtask_1', 'answer of stage_2.subtask_1']
    }
    results10, log10 = await self.cot(
        subtask_id='stage_2.subtask_2',
        cot_agent_desc=cot_instruction_2
    )
    logs.append(log10)

    final_answer = await self.make_final_answer(results10['thinking'], results10['answer'])
    return final_answer, logs
