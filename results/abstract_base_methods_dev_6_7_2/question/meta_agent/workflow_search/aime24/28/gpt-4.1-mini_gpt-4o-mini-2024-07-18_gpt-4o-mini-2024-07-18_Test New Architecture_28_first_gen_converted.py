async def forward_28(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction2 = "Subtask 2: Derive the numerical values of the torus major radius (6), minor radius (3), and sphere radius (11) from the problem statement."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, deriving radii, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    debate_instruction1 = "Subtask 1: Select the derived values as the key geometric parameters for further analysis, debating their correctness and relevance."
    debate_desc1 = {
        'instruction': debate_instruction1,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc1,
        final_decision_desc={
            'instruction': "Subtask 1: Make final decision on selected geometric parameters.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results1['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating parameters, thinking: {results1['list_thinking'][round][idx].content}; answer: {results1['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding parameters, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction0 = "Subtask 0: Evaluate and validate the input parameters for consistency and correctness based on selected parameters from Subtask 1 and derived values from Subtask 2."
    cot_sc_desc0 = {
        'instruction': cot_sc_instruction0,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results0 = await self.sc_cot(
        subtask_id="subtask_0",
        cot_sc_desc=cot_sc_desc0,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results0['list_thinking']):
        agents.append(f"SC-CoT agent {results0['cot_agent'][idx].id}, validating parameters, thinking: {results0['list_thinking'][idx]}; answer: {results0['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 0 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    loop_subtask_ids = [3]
    for i in loop_subtask_ids:
        cot_instruction3 = "Subtask 3: Generate the implicit equations for torus-sphere tangency in both inner and outer-surface cases, using consolidated parameters."
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [taskInfo, results0['thinking'], results0['answer']],
            'temperature': 0.0,
            'context': ["user query", "validated parameters"]
        }
        results3 = await self.cot(
            subtask_id="subtask_3",
            cot_agent_desc=cot_agent_desc3
        )
        agents.append(f"CoT agent {results3['cot_agent'].id}, generating tangency equations, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])

    aggregate_instruction4 = "Subtask 4: Consolidate the two tangency-case equations into a unified framework to solve for the contact circle radii."
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "tangency equations"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating equations, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    condition = True
    if condition:
        cot_answergen_instruction5 = "Subtask 5: Identify and fill any missing derivation steps or logical gaps in the consolidated solution to ensure completeness."
        cot_answergen_desc5 = {
            'instruction': cot_answergen_instruction5,
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'temperature': 0.0,
            'context': ["user query", "consolidated solution"]
        }
        results5 = await self.answer_generate(
            subtask_id="subtask_5",
            cot_agent_desc=cot_answergen_desc5
        )
        agents.append(f"AnswerGenerate agent {results5['cot_agent'].id}, filling gaps, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
        sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
        logs.append(results5['subtask_desc'])
    else:
        formatter_instruction6 = "Subtask 6: Identify, clarify, and validate unit consistency across all distance values in the derivation."
        formatter_desc6 = {
            'instruction': formatter_instruction6,
            'input': [taskInfo, results4['thinking'], results4['answer']],
            'temperature': 0.0,
            'context': ["user query", "consolidated solution"],
            'format': 'short and concise, without explanation'
        }
        results6 = await self.specific_format(
            subtask_id="subtask_6",
            formatter_desc=formatter_desc6
        )
        agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, validating units, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])

    review_instruction7 = "Subtask 7: Validate the final answer against problem requirements and check for completeness."
    review_desc7 = {
        'instruction': review_instruction7,
        'input': [taskInfo, results5['thinking'] if condition else results6['thinking'], results5['answer'] if condition else results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "final solution"]
    }
    results7 = await self.review(
        subtask_id="subtask_7",
        review_desc=review_desc7
    )
    agents.append(f"Review agent {results7['review_agent'].id}, validating final answer, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    formatter_instruction8 = "Subtask 8: Enhance clarity and coherence of the full solution write-up."
    formatter_desc8 = {
        'instruction': formatter_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "validated final answer"],
        'format': 'short and concise, without explanation'
    }
    results8 = await self.specific_format(
        subtask_id="subtask_8",
        formatter_desc=formatter_desc8
    )
    agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, enhancing clarity, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    review_instruction9 = "Subtask 9: Validate the final answer against problem requirements and check for completeness."
    review_desc9 = {
        'instruction': review_instruction9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "clarified final answer"]
    }
    results9 = await self.review(
        subtask_id="subtask_9",
        review_desc=review_desc9
    )
    agents.append(f"Review agent {results9['review_agent'].id}, final validation, feedback: {results9['thinking'].content}; correct: {results9['answer'].content}")
    sub_tasks.append(f"Subtask 9 output: feedback - {results9['thinking'].content}; correct - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    formatter_instruction10 = "Subtask 10: Format the final computed m+n result as a clean integer output."
    formatter_desc10 = {
        'instruction': formatter_instruction10,
        'input': [taskInfo, results9['thinking'], results9['answer']],
        'temperature': 0.0,
        'context': ["user query", "validated final integer result"],
        'format': 'short and concise, without explanation'
    }
    results10 = await self.specific_format(
        subtask_id="subtask_10",
        formatter_desc=formatter_desc10
    )
    agents.append(f"SpecificFormat agent {results10['formatter_agent'].id}, formatting final integer, thinking: {results10['thinking'].content}; answer: {results10['answer'].content}")
    sub_tasks.append(f"Subtask 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
    logs.append(results10['subtask_desc'])

    final_answer = await self.make_final_answer(results10['thinking'], results10['answer'], sub_tasks, agents)
    return final_answer, logs
