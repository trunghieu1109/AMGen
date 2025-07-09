async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Analyze the problem statement to identify that we need the least prime p such that there exists n with p^2 dividing n^4 + 1, and find the least m satisfying m^4 + 1 ≡ 0 mod p^2."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing problem statement, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_sc_instruction2 = "Subtask 2: Derive the divisibility conditions for n^4 ≡ -1 mod p^2 using Hensel's lemma or lifting arguments based on output from Subtask 1."
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
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, deriving divisibility conditions, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    debate_instruction3 = "Subtask 3: Select candidate primes p by evaluating the criterion that p divides n^4 + 1 for some n, e.g., p ≡ 1 mod 4, and shortlist primes to test based on output from Subtask 2."
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
            agents.append(f"Debate agent {agent.id}, round {round}, selecting candidate primes, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    candidate_primes = []
    if 'answer' in results3 and results3['answer']:
        candidate_primes_text = results3['answer'].content
        import re
        candidate_primes = list(map(int, re.findall(r'\b\d+\b', candidate_primes_text)))
    results_loop = []
    for idx, p in enumerate(candidate_primes):
        cot_instruction4 = f"Subtask 4: For candidate prime p={p}, attempt to find integer n such that p^2 divides n^4 + 1 using derived conditions."
        cot_agent_desc4 = {
            'instruction': cot_instruction4,
            'input': [taskInfo, str(p)],
            'temperature': 0.0,
            'context': ["user query", f"candidate prime {p}"]
        }
        results4 = await self.answer_generate(
            subtask_id=f"subtask_4_{idx+1}",
            cot_agent_desc=cot_agent_desc4
        )
        agents.append(f"AnswerGenerate agent {results4['cot_agent'].id}, testing prime {p}, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Subtask 4_{idx+1} output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])
        results_loop.append((p, results4))
    aggregate_instruction5 = "Subtask 5: Aggregate results from candidate primes testing, identify the least prime p for which p^2 divides n^4 + 1, find the least positive integer m such that m^4 + 1 ≡ 0 mod p^2, validate correctness, and format the final numeric answer."
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo] + [r[1]['answer'] for r in results_loop],
        'temperature': 0.0,
        'context': ["user query", "candidate primes testing results"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc5
    )
    agents.append(f"Aggregate agent {results5['aggregate_agent'].id}, aggregating candidate results, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    specific_format_instruction = "Subtask 6: Format the final numeric answer as an integer only, without explanation."
    specific_format_desc = {
        'instruction': specific_format_instruction,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "final aggregation result"],
        'format': 'short and concise, without explanation'
    }
    results6 = await self.specific_format(
        subtask_id="subtask_6",
        formatter_desc=specific_format_desc
    )
    agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, formatting final answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
