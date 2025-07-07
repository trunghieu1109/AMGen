async def forward_4(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Identify and isolate the key elements of the problem: the prime p, integers n and m, and the divisibility conditions p²│(n⁴+1) and p²│(m⁴+1)."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing key elements, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Analyze and characterize the condition n⁴+1 ≡ 0 mod p² to derive constraints and methods for testing primes."
    cot_sc_desc = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, analyzing divisibility condition, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    candidate_primes = []
    for i in range(2, 1000):
        cot_answer_generate_instruction = f"Subtask 3: Generate candidate prime p = {i} and search for integer n such that p² divides n⁴+1."
        cot_answer_generate_desc = {
            'instruction': cot_answer_generate_instruction,
            'input': [taskInfo, results2['thinking'], results2['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
        }
        results3 = await self.answer_generate(
            subtask_id=f"subtask_3_{i}",
            cot_agent_desc=cot_answer_generate_desc
        )
        agents.append(f"AnswerGenerate agent {results3['cot_agent'].id}, generating candidate prime {i}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        if 'valid prime' in results3['answer'].content.lower():
            candidate_primes.append(i)
            break
    
    aggregate_instruction = "Subtask 4: Aggregate all primes found that satisfy the condition and select the smallest such prime p."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_primes,
        'temperature': 0.0,
        'context': ["user query", "candidate primes"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, aggregating candidate primes, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    answer_generate_instruction = "Subtask 5: Validate that the selected prime p is indeed the least prime for which there exists n with p²│(n⁴+1)."
    answer_generate_desc = {
        'instruction': answer_generate_instruction,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.answer_generate(
        subtask_id="subtask_5",
        cot_agent_desc=answer_generate_desc
    )
    agents.append(f"AnswerGenerate agent {results5['cot_agent'].id}, validating least prime, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    condition_met = 'true' in results5['answer'].content.lower() or 'valid' in results5['answer'].content.lower()

    if condition_met:
        # True branch: find least positive integer m such that p² divides m⁴+1
        # Stage 5
        specific_format_instruction = "Subtask 6: Identify discrete steps to compute the least positive integer m such that p²│(m⁴+1)."
        specific_format_desc = {
            'instruction': specific_format_instruction,
            'input': [taskInfo, results4['answer']],
            'temperature': 0.0,
            'context': ["user query", "least prime p"]
        }
        results6 = await self.specific_format(
            subtask_id="subtask_6",
            formatter_desc=specific_format_desc
        )
        agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, identifying steps for m, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])

        answer_generate_instruction2 = "Subtask 7: Generate candidate m and verify p²│(m⁴+1)."
        answer_generate_desc2 = {
            'instruction': answer_generate_instruction2,
            'input': [taskInfo, results4['answer'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "least prime p", "steps for m"]
        }
        results7 = await self.answer_generate(
            subtask_id="subtask_7",
            cot_agent_desc=answer_generate_desc2
        )
        agents.append(f"AnswerGenerate agent {results7['cot_agent'].id}, computing and validating m, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
        sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        review_instruction = "Subtask 8: Evaluate and validate the computed m against the divisibility criterion p²│(m⁴+1) to ensure correctness."
        review_desc = {
            'instruction': review_instruction,
            'input': [taskInfo, results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
        }
        results8 = await self.review(
            subtask_id="subtask_8",
            review_desc=review_desc
        )
        agents.append(f"Review agent {results8['review_agent'].id}, validating m, feedback: {results8['thinking'].content}; correct: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: feedback - {results8['thinking'].content}; correct - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        # Stage 7
        revise_instruction = "Subtask 9: Refine and enhance the clarity and coherence of the solution presentation."
        revise_desc = {
            'instruction': revise_instruction,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'temperature': 0.0,
            'context': ["user query", "feedback of subtask 8", "correctness of subtask 8"]
        }
        results9 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=revise_desc
        )
        agents.append(f"SpecificFormat agent {results9['formatter_agent'].id}, refining solution clarity, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        review_instruction2 = "Subtask 10: Review the refined solution for coherence and correctness."
        review_desc2 = {
            'instruction': review_instruction2,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
        }
        results10 = await self.review(
            subtask_id="subtask_10",
            review_desc=review_desc2
        )
        agents.append(f"Review agent {results10['review_agent'].id}, reviewing refined solution, feedback: {results10['thinking'].content}; correct: {results10['answer'].content}")
        sub_tasks.append(f"Subtask 10 output: feedback - {results10['thinking'].content}; correct - {results10['answer'].content}")
        logs.append(results10['subtask_desc'])

        # Stage 8
        format_instruction = "Subtask 11: Format the final result into the required concise integer output (the value of m)."
        format_desc = {
            'instruction': format_instruction,
            'input': [taskInfo, results10['thinking'], results10['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 10", "answer of subtask 10"],
            'format': 'short and concise, without explanation'
        }
        results11 = await self.specific_format(
            subtask_id="subtask_11",
            formatter_desc=format_desc
        )
        agents.append(f"SpecificFormat agent {results11['formatter_agent'].id}, formatting final output, thinking: {results11['thinking'].content}; answer: {results11['answer'].content}")
        sub_tasks.append(f"Subtask 11 output: thinking - {results11['thinking'].content}; answer - {results11['answer'].content}")
        logs.append(results11['subtask_desc'])

        final_answer = await self.make_final_answer(results11['thinking'], results11['answer'], sub_tasks, agents)
        return final_answer, logs

    else:
        # False branch: no valid prime found or validation failed
        final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
        return final_answer, logs
