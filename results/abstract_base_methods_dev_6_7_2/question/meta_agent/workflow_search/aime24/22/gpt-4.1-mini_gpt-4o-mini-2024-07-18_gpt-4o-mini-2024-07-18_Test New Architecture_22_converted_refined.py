async def forward_22(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction0 = "Subtask 1: Enumerate and justify all feasible list lengths and configurations that satisfy the sum=30 and median constraints, including explicit examples and reasoning steps."
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc0
    )
    agents.append(f"CoT agent {results0['cot_agent'].id}, enumerating feasible list lengths and configurations, thinking: {results0['thinking'].content}; answer: {results0['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    cot_reflect_instruction1 = "Subtask 2: Reflect on the derivation of the median position and possible median values given list length n, ensuring the median is a positive integer not in the list and correctly calculated for odd and even lengths."
    critic_instruction1 = "Please review the median calculation and its consistency with the constraints, providing feedback and corrections if needed."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_2",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, reflecting on median calculation, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction2 = "Subtask 3: Define problem variables including list length n, list elements x_i, and restate all constraints: sum=30, unique mode=9, median positive integer not in list."
    cot_sc_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, defining variables and constraints, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    sc_cot_instruction3 = "Subtask 4: Generate multiple candidate lists independently that satisfy sum=30, unique mode=9, and median positive integer not in the list, cross-validating medians and modes for correctness."
    sc_cot_desc3 = {
        'instruction': sc_cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.7,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_desc=sc_cot_desc3,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, generating candidate lists, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction4 = "Subtask 5: Critically evaluate and debate the validity of candidate lists generated, challenging each candidate's adherence to sum, unique mode, and median constraints, and justify the selection of the most reliable solutions."
    final_decision_instruction4 = "Subtask 5: Make final decision on the most valid candidate lists after debate."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc4 = {
        'instruction': final_decision_instruction4,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc4,
        final_decision_desc=final_decision_desc4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating candidate validity, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding valid candidates, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction5 = "Subtask 6: Reflect on and validate the candidate lists selected from debate, reviewing validation logic for sum, unique mode, and median constraints, and test edge cases to ensure correctness."
    critic_instruction5 = "Please review the validation logic and provide feedback on any logical errors or edge cases missed."
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    critic_desc5 = {
        'instruction': critic_instruction5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc5,
        critic_desc=critic_desc5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, validating candidate lists, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correct: {results5['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_reflect_instruction6 = "Subtask 7: Reflect on the final candidate list(s) to verify all constraints including sum=30, unique mode=9, median positive integer not in list, before computing the sum of squares."
    critic_instruction6 = "Please review the final candidate list(s) for correctness and completeness of constraints."
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    critic_desc6 = {
        'instruction': critic_instruction6,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_7",
        cot_reflect_desc=cot_reflect_desc6,
        critic_desc=critic_desc6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, verifying final candidate list, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][i].content}; correct: {results6['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_reflect_instruction7 = "Subtask 8: Reflect on the calculation of the sum of squares of the final candidate list, ensuring correctness and consistency with all constraints."
    critic_instruction7 = "Please review the sum of squares calculation and confirm it is based on a valid candidate list."
    cot_reflect_desc7 = {
        'instruction': cot_reflect_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    critic_desc7 = {
        'instruction': critic_instruction7,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results7 = await self.reflexion(
        subtask_id="subtask_8",
        cot_reflect_desc=cot_reflect_desc7,
        critic_desc=critic_desc7,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, reflecting on sum of squares calculation, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, feedback: {results7['list_feedback'][i].content}; correct: {results7['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    specific_format_instruction9 = "Subtask 9: Format the verified final answer (sum of squares) into the specified JSON structure as required by the query."
    specific_format_desc9 = {
        'instruction': specific_format_instruction9,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "final sum of squares answer"],
        'format': 'JSON structure as required'
    }
    results9 = await self.specific_format(
        subtask_id="subtask_9",
        formatter_desc=specific_format_desc9
    )

    answer_generate_instruction9 = "Subtask 10: Generate the final answer in the required format based on the formatted JSON structure."
    answer_generate_desc9 = {
        'instruction': answer_generate_instruction9,
        'input': [taskInfo, results9['thinking'], results9['answer']],
        'temperature': 0.0,
        'context': ["user query", "formatted final answer"]
    }
    results_final = await self.answer_generate(
        subtask_id="subtask_10",
        cot_agent_desc=answer_generate_desc9
    )
    agents.append(f"AnswerGenerate agent {results_final['cot_agent'].id}, generating final answer, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs
