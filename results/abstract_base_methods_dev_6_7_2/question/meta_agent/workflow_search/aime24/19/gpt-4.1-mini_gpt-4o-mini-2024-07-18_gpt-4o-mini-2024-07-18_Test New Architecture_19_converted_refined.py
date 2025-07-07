async def forward_19(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instruction_1 = (
        "Sub-task 1: Debate the evaluation of the product \u220f_{k=0}^{12} (2 - 2\u03c9^k + \u03c9^{2k}) where \u03c9 \neq 1 is a 13th root of unity. "
        "Explicitly consider the term at k=0 and discuss whether it should be included or excluded in the product. "
        "Analyze the expression using properties of roots of unity and identify any zero factors or simplifications."
    )
    debate_desc_1 = {
        'instruction': debate_instruction_1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    final_decision_desc_1 = {
        'instruction': "Sub-task 1: Make final decision on the correct evaluation and domain of the product, clarifying the role of k=0.",
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results1 = await self.debate(
        subtask_id="subtask_1",
        debate_desc=debate_desc_1,
        final_decision_desc=final_decision_desc_1,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results1['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, evaluating product factors, thinking: {results1['list_thinking'][round][idx].content}; answer: {results1['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, evaluating product factors, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_reflect_instruction_2 = (
        "Sub-task 2: Reflect on the debate conclusion and carefully re-derive the product \u220f_{k=0}^{12} (2 - 2\u03c9^k + \u03c9^{2k}), "
        "validating the zero factor at k=0 and using cyclotomic polynomial properties to express the product over nontrivial roots. "
        "Check each factor and confirm the correct unified expression."
    )
    critic_instruction_2 = "Please review the re-derivation for correctness, especially handling the zero factor and product domain."
    cot_reflect_desc_2 = {
        'instruction': cot_reflect_instruction_2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc_2 = {
        'instruction': critic_instruction_2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc_2,
        critic_desc=critic_desc_2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, re-deriving product with zero factor consideration, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, providing feedback, thinking: {results2['list_feedback'][i].content}; answer: {results2['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction_3 = (
        "Sub-task 3: Using self-consistency CoT, explore multiple algebraic transformations to simplify each term (2 - 2\u03c9^k + \u03c9^{2k}) "
        "for k=1 to 12, ensuring correct application of roots of unity properties and excluding k=0. "
        "Vote on the most accurate simplification among the reasoning paths."
    )
    N = self.max_sc
    cot_sc_desc_3 = {
        'instruction': cot_sc_instruction_3,
        'input': [taskInfo, results2['thinking'], results2['answer'], results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc_3,
        n_repeat=N
    )
    for idx in range(len(results3['list_thinking'])):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, exploring algebraic transformations, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    candidate_expressions = []
    for k in range(13):
        debate_instruction_4 = (
            f"Sub-task 4: Debate candidate expressions for the factor at k={k} using the simplified forms from Sub-task 3. "
            "Explicitly discuss the zero factor at k=0 and its impact on the product. "
            "Determine the correct expression for each k."
        )
        debate_desc_4 = {
            'instruction': debate_instruction_4,
            'input': [taskInfo, results3['thinking'], results3['answer'], k],
            'temperature': 0.5,
            'context': ["user query", f"simplified forms from subtask 3", f"k={k}"]
        }
        final_decision_desc_4 = {
            'instruction': f"Sub-task 4: Make final decision on candidate expression for k={k}.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        }
        results4 = await self.debate(
            subtask_id=f"subtask_4_{k+1}",
            debate_desc=debate_desc_4,
            final_decision_desc=final_decision_desc_4,
            n_repeat=self.max_round
        )
        for round in range(self.max_round):
            for idx, agent in enumerate(results4['debate_agent']):
                agents.append(f"Debate agent {agent.id}, round {round}, candidate expression k={k}, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
        agents.append(f"Final Decision agent, candidate expression k={k}, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Sub-task 4_{k+1} output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])
        candidate_expressions.append((results4['thinking'], results4['answer']))

    cot_sc_instruction_5 = (
        "Sub-task 5: Using self-consistency CoT, aggregate candidate expressions from Sub-task 4 to derive a consistent closed-form product. "
        "Explore multiple algebraic derivations and vote on the most consistent product value, reconciling conflicting results such as 169 vs 13."
    )
    cot_sc_desc_5 = {
        'instruction': cot_sc_instruction_5,
        'input': [taskInfo] + candidate_expressions,
        'temperature': 0.5,
        'context': ["user query", "candidate expressions from subtask 4"]
    }
    results5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_desc=cot_sc_desc_5,
        n_repeat=N
    )
    for idx in range(len(results5['list_thinking'])):
        agents.append(f"CoT-SC agent {results5['cot_agent'][idx].id}, aggregating candidates, thinking: {results5['list_thinking'][idx]}; answer: {results5['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    debate_instruction_6 = (
        "Sub-task 6: Debate to identify and fill any deficiencies or missing algebraic justifications in the aggregated derivation from Sub-task 5. "
        "Verify modular arithmetic correctness and completeness of the derivation."
    )
    debate_desc_6 = {
        'instruction': debate_instruction_6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    final_decision_desc_6 = {
        'instruction': "Sub-task 6: Make final decision on completeness and correctness of the derivation and modular arithmetic.",
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=debate_desc_6,
        critic_desc=final_decision_desc_6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, filling gaps and verifying modular arithmetic, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, providing feedback, thinking: {results6['list_feedback'][i].content}; answer: {results6['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_instruction_7 = (
        "Sub-task 7: Provide a clear and coherent summary of the algebraic reasoning and derivation leading to the final answer for the product remainder modulo 1000."
    )
    cot_agent_desc_7 = {
        'instruction': cot_instruction_7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7 = await self.cot(
        subtask_id="subtask_7",
        cot_agent_desc=cot_agent_desc_7
    )
    agents.append(f"CoT agent {results7['cot_agent'].id}, summarizing algebraic reasoning, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_reflect_instruction_8 = (
        "Sub-task 8: Reflectively validate the final result by detailed algebraic proof and modular arithmetic checks, ensuring consistency and correctness."
    )
    critic_instruction_8 = "Please review the validation steps and provide any corrections or confirmations."
    cot_reflect_desc_8 = {
        'instruction': cot_reflect_instruction_8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    critic_desc_8 = {
        'instruction': critic_instruction_8,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results8 = await self.reflexion(
        subtask_id="subtask_8",
        cot_reflect_desc=cot_reflect_desc_8,
        critic_desc=critic_desc_8,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results8['cot_agent'].id}, validating final result, thinking: {results8['list_thinking'][0].content}; answer: {results8['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results8['list_feedback']))):
        agents.append(f"Critic agent {results8['critic_agent'].id}, providing feedback, thinking: {results8['list_feedback'][i].content}; answer: {results8['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    cot_instruction_9 = (
        "Sub-task 9: Explicitly identify, clarify, and validate discrete reasoning units in the final answer ensuring all correctness criteria are met."
    )
    cot_agent_desc_9 = {
        'instruction': cot_instruction_9,
        'input': [taskInfo, results8['thinking'], results8['answer'], results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 8", "answer of subtask 8", "thinking of subtask 7", "answer of subtask 7"]
    }
    results9 = await self.cot(
        subtask_id="subtask_9",
        cot_agent_desc=cot_agent_desc_9
    )
    agents.append(f"CoT agent {results9['cot_agent'].id}, validating reasoning units, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    cot_instruction_10 = (
        "Sub-task 10: Format the final remainder result as an integer modulo 1000 in the specified concise output format."
    )
    cot_agent_desc_10 = {
        'instruction': cot_instruction_10,
        'input': [taskInfo, results9['thinking'], results9['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 9", "answer of subtask 9"],
        'format': 'short and concise, without explaination'
    }
    results10 = await self.cot(
        subtask_id="subtask_10",
        cot_agent_desc=cot_agent_desc_10
    )
    agents.append(f"CoT agent {results10['cot_agent'].id}, formatting final answer, thinking: {results10['thinking'].content}; answer: {results10['answer'].content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
    logs.append(results10['subtask_desc'])

    final_answer_processed = await self.make_final_answer(results10['thinking'], results10['answer'], sub_tasks, agents)
    return final_answer_processed, logs
