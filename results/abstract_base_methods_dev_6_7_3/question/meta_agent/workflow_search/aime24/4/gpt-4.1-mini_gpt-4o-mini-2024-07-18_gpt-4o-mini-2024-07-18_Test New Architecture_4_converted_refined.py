async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = "Subtask 1: Carefully analyze the problem statement to identify the least prime p such that there exists a positive integer n with p^2 dividing n^4+1, and clarify all constraints. Validate your understanding by checking consistency with the problem."
    critic_instruction1 = "Please review the analysis and provide feedback on any missing or incorrect interpretations."
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, analyzing problem statement, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining analysis, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Systematically explore primes in ascending order and analyze the condition n^4 + 1 ≡ 0 (mod p^2) using modular arithmetic. For each prime, consider multiple reasoning paths to verify if such n exists."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx in range(len(results2['list_thinking'])):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, exploring primes and modular conditions, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    candidate_primes = []
    for p in range(2, 1000):
        cot_reflect_instruction3 = f"Subtask 3: For prime p = {p}, reflectively enumerate integers n from 0 to p-1, compute n^4 + 1 mod p^2, and determine if any n satisfies the divisibility condition. Stop once a valid n is found or p is ruled out."
        critic_instruction3 = f"Please review the enumeration and modular computations for p = {p} and provide feedback on correctness and completeness."
        cot_reflect_desc3 = {
            'instruction': cot_reflect_instruction3,
            'input': [taskInfo, results2['thinking'], results2['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        critic_desc3 = {
            'instruction': critic_instruction3,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results3 = await self.reflexion(
            subtask_id=f"subtask_3_{p}",
            cot_reflect_desc=cot_reflect_desc3,
            critic_desc=critic_desc3,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, checking prime {p}, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results3['list_feedback']))):
            agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correct: {results3['list_correct'][i].content}")
            if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
                agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining check for prime {p}, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
        sub_tasks.append(f"Sub-task 3 output for prime {p}: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        if 'valid prime' in results3['answer'].content.lower() or 'found n' in results3['answer'].content.lower():
            candidate_primes.append({'prime': p, 'witness_n': results3['answer'].content})
            break

    aggregate_instruction4 = "Subtask 4: Aggregate all primes for which a valid n was found such that p^2 divides n^4+1, and select the smallest such prime p by consensus among multiple agents."
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo] + [cp['prime'] for cp in candidate_primes],
        'temperature': 0.0,
        'context': ["user query", "candidate primes with witnesses"]
    }
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_desc=aggregate_desc4,
        n_repeat=self.max_sc
    )
    for idx in range(len(results4['list_thinking'])):
        agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, aggregating candidate primes, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction5 = "Subtask 5: Validate that no prime smaller than the selected prime p admits an integer n with p^2 dividing n^4+1. Then explicitly find the least positive integer m such that p^2 divides m^4+1."
    critic_instruction5 = "Please review the validation and final computation of m, providing feedback on correctness and completeness."
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc5 = {
        'instruction': critic_instruction5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc5,
        critic_desc=critic_desc5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, validating least prime and computing m, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correct: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final validation and computation, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
