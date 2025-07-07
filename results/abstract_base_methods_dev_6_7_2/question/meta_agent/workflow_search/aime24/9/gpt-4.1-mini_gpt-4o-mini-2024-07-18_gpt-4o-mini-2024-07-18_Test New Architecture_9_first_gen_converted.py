async def forward_9(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Compute total number of lottery combinations and counts of outcomes satisfying at least two matches, with context from the lottery problem description."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing total combinations and counts, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Sub-task 2: Express counts as probabilities by dividing by total combinations, using outputs from Sub-task 1."
    N = self.max_sc
    cot_sc_desc = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc,
        n_repeat=N
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, consider probabilities, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    logs.append(results2['subtask_desc'])

    debate_instruction3 = "Sub-task 3: Select combinatorial counts for each match category and simplify the final ratio m/n, then compute m+n, based on outputs from Sub-task 2."
    final_decision_instruction3 = "Sub-task 3: Make final decision on simplified ratio and sum m+n."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc3 = {
        'instruction': final_decision_instruction3,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, simplifying ratio and computing m+n, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating final m+n, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    results_loop = []
    for k in [2, 3, 4]:
        answergen_instruction = f"Sub-task 4: For k={k}, compute C(4,k)*C(6,4-k) to count outcomes with exactly k matches."
        answergen_desc = {
            'instruction': answergen_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results_loop_k = await self.answer_generate(
            subtask_id=f"subtask_{4+k-2}",
            cot_agent_desc=answergen_desc
        )
        agents.append(f"AnswerGenerate agent {results_loop_k['cot_agent'].id}, computing count for k={k}, thinking: {results_loop_k['thinking'].content}; answer: {results_loop_k['answer'].content}")
        sub_tasks.append(f"Sub-task {4+k-2} output: thinking - {results_loop_k['thinking'].content}; answer - {results_loop_k['answer'].content}")
        logs.append(results_loop_k['subtask_desc'])
        results_loop.append(results_loop_k)

    aggregate_instruction = "Sub-task 7: Sum the counts from each k=2,3,4 to get total prize-winning outcomes."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + [r['answer'] for r in results_loop],
        'temperature': 0.0,
        'context': ["user query", "counts for k=2,3,4"]
    }
    results7 = await self.aggregate(
        subtask_id="subtask_7",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results7['aggregate_agent'].id}, summing counts, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    answergen_instruction8 = "Sub-task 8: Identify any missing steps (e.g., probability of grand prize, conditional step) and add them."
    answergen_desc8 = {
        'instruction': answergen_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results8 = await self.answer_generate(
        subtask_id="subtask_8",
        cot_agent_desc=answergen_desc8
    )
    agents.append(f"AnswerGenerate agent {results8['cot_agent'].id}, identifying missing steps, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    condition = True
    if condition:
        cot_reflect_instruction9 = "Sub-task 9: Extract and validate the final m and n values to ensure they represent the simplified ratio."
        critic_instruction9 = "Please review the extracted m and n values for correctness and simplification."
        cot_reflect_desc9 = {
            'instruction': cot_reflect_instruction9,
            'input': [taskInfo, results3['thinking'], results3['answer'], results8['thinking'], results8['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 8", "answer of subtask 8"]
        }
        critic_desc9 = {
            'instruction': critic_instruction9,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results9 = await self.reflexion(
            subtask_id="subtask_9",
            cot_reflect_desc=cot_reflect_desc9,
            critic_desc=critic_desc9,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results9['cot_agent'].id}, validating m and n, thinking: {results9['list_thinking'][0].content}; answer: {results9['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results9['list_feedback']))):
            agents.append(f"Critic agent {results9['critic_agent'].id}, feedback: {results9['list_feedback'][i].content}; correct: {results9['list_correct'][i].content}")
        sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])
    else:
        review_instruction10 = "Sub-task 10: Validate the computed m+n against criteria and confirm numeric correctness."
        review_desc10 = {
            'instruction': review_instruction10,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results10 = await self.review(
            subtask_id="subtask_10",
            review_desc=review_desc10
        )
        agents.append(f"Review agent {results10['review_agent'].id}, validating m+n, feedback: {results10['thinking'].content}; correct: {results10['answer'].content}")
        sub_tasks.append(f"Sub-task 10 output: feedback - {results10['thinking'].content}; correct - {results10['answer'].content}")
        logs.append(results10['subtask_desc'])

    review_instruction11 = "Sub-task 11: Refine the explanation and ensure the presentation of m+n is clear and coherent."
    review_desc11 = {
        'instruction': review_instruction11,
        'input': [taskInfo, results9['thinking'], results9['answer']],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results11 = await self.review(
        subtask_id="subtask_11",
        review_desc=review_desc11
    )
    agents.append(f"Review agent {results11['review_agent'].id}, refining explanation, feedback: {results11['thinking'].content}; correct: {results11['answer'].content}")
    sub_tasks.append(f"Sub-task 11 output: feedback - {results11['thinking'].content}; correct - {results11['answer'].content}")
    logs.append(results11['subtask_desc'])

    formatter_instruction12 = "Sub-task 12: Format the final answer (m+n) as a single integer output."
    formatter_desc12 = {
        'instruction': formatter_instruction12,
        'input': [taskInfo, results9['thinking'], results9['answer']],
        'temperature': 0.0,
        'context': ["user query"],
        'format': 'short and concise, without explanation'
    }
    results12 = await self.specific_format(
        subtask_id="subtask_12",
        formatter_desc=formatter_desc12
    )
    agents.append(f"SpecificFormat agent {results12['formatter_agent'].id}, formatting final answer, thinking: {results12['thinking'].content}; answer: {results12['answer'].content}")
    sub_tasks.append(f"Sub-task 12 output: thinking - {results12['thinking'].content}; answer - {results12['answer'].content}")
    logs.append(results12['subtask_desc'])

    final_answer = await self.make_final_answer(results12['thinking'], results12['answer'], sub_tasks, agents)
    return final_answer, logs
