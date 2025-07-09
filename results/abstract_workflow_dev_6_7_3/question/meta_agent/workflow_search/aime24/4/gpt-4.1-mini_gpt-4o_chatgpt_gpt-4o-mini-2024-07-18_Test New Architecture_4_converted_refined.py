async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_reflect_instruction1 = "Subtask 1: Analyze the problem to find the least prime p such that there exists a positive integer n with p^2 dividing n^4 + 1. Verify modular arithmetic computations step-by-step and reflect on correctness of intermediate results."
    critic_instruction1 = "Please review the modular arithmetic computations and reasoning steps for correctness and identify any errors or assumptions."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, analyzing least prime p, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correctness: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining answer, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_instruction2 = "Subtask 2: Validate the base solution modulo p^2 found in Subtask 1 and carefully apply Hensel's lemma or lifting arguments to derive divisibility conditions for n^4 + 1 modulo p^2."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, validating base solution and applying lifting, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    debate_instruction3 = "Subtask 3: Debate among multiple agents to propose and test candidate primes p (starting from smallest primes congruent to 1 mod 4) for which p^2 divides n^4 + 1 for some n, challenging assumptions about minimality and correctness."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=None,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, testing candidate primes, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    candidate_primes = []
    if 'answer' in results3 and results3['answer']:
        import re
        candidate_primes = list(map(int, re.findall(r'\b\d+\b', results3['answer'].content)))
    results_loop = []
    for idx, p in enumerate(candidate_primes):
        cot_sc_instruction4 = f"Subtask 4: For candidate prime p={p}, find integer n such that p^2 divides n^4 + 1, and find the least positive integer m such that m^4 + 1 ≡ 0 mod p^2, using self-consistency to explore multiple reasoning paths."
        cot_sc_desc4 = {
            'instruction': cot_sc_instruction4,
            'input': [taskInfo, str(p)],
            'temperature': 0.5,
            'context': ["user query", f"candidate prime {p}"]
        }
        results4 = await self.sc_cot(
            subtask_id=f"subtask_4_{idx+1}",
            cot_sc_desc=cot_sc_desc4,
            n_repeat=self.max_sc
        )
        for i in range(self.max_sc):
            agents.append(f"CoT-SC agent {results4['cot_agent'][i].id}, candidate prime {p}, reasoning path {i}, thinking: {results4['list_thinking'][i]}; answer: {results4['list_answer'][i]}")
        sub_tasks.append(f"Subtask 4_{idx+1} output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])
        results_loop.append((p, results4))
    aggregate_instruction5 = "Subtask 5: Aggregate and cross-validate results from candidate primes and their corresponding least positive integers m, using self-consistency to identify the consistent and correct minimal prime p and integer m satisfying the divisibility condition."
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo] + [r[1]['answer'] for r in results_loop],
        'temperature': 0.5,
        'context': ["user query", "candidate primes testing results"]
    }
    results5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_desc=aggregate_desc5,
        n_repeat=self.max_sc
    )
    for i in range(self.max_sc):
        agents.append(f"CoT-SC agent {results5['cot_agent'][i].id}, aggregating candidate results, reasoning path {i}, thinking: {results5['list_thinking'][i]}; answer: {results5['list_answer'][i]}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    cot_reflect_instruction6 = "Subtask 6: Reflect on the aggregated final answer to confirm correctness and consistency with problem requirements before formatting the output."
    critic_instruction6 = "Please review the final aggregated answer for correctness and identify any remaining issues or confirm validity."
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "aggregated final answer"]
    }
    critic_desc6 = {
        'instruction': critic_instruction6,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc6,
        critic_desc=critic_desc6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, final validation, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][i].content}; correctness: {results6['list_correct'][i].content}")
        if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
            agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final answer, thinking: {results6['list_thinking'][i + 1].content}; answer: {results6['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    specific_format_instruction = "Subtask 7: Format the final numeric answer as an integer only, without explanation."
    specific_format_desc = {
        'instruction': specific_format_instruction,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "final validated answer"],
        'format': 'short and concise, without explanation'
    }
    results7 = await self.specific_format(
        subtask_id="subtask_7",
        formatter_desc=specific_format_desc
    )
    agents.append(f"SpecificFormat agent {results7['formatter_agent'].id}, formatting final answer, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
