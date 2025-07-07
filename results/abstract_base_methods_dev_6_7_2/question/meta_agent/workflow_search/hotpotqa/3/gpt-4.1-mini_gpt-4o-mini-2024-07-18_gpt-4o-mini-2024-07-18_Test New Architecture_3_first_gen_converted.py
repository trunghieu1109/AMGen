async def forward_3(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction1 = "Subtask 1: Identify the creator of 'Wallace and Gromit' with detailed reasoning and context from the taskInfo."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing creator identification, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_ag_instruction2 = "Subtask 2: List other animation comedies created by the identified creator from Subtask 1, with reasoning and answer generation."
    cot_ag_desc2 = {
        'instruction': cot_ag_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.answer_generate(
        subtask_id="subtask_2",
        cot_agent_desc=cot_ag_desc2
    )
    agents.append(f"CoT + AnswerGenerate agent {results2['cot_agent'].id}, listing animation comedies, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    cot_ag_instruction3 = "Subtask 3: From the list of animation comedies in Subtask 2, identify the one that matches the description of featuring animated zoo animals with a soundtrack of people talking about their homes, with reasoning and answer generation."
    cot_ag_desc3 = {
        'instruction': cot_ag_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.answer_generate(
        subtask_id="subtask_3",
        cot_agent_desc=cot_ag_desc3
    )
    agents.append(f"CoT + AnswerGenerate agent {results3['cot_agent'].id}, identifying matching animation comedy, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    aggregate_instruction4 = "Subtask 4: Consolidate the candidate outputs from Subtask 3 to synthesize a single coherent answer identifying the animation comedy."
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 3"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating candidate outputs, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    review_instruction5 = "Subtask 5: Validate the consolidated answer from Subtask 4 to ensure it accurately and completely addresses the query, using review and CoT reasoning."
    review_desc5 = {
        'instruction': review_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc5
    )
    agents.append(f"Review agent {results5['review_agent'].id}, reviewing consolidated answer, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    programmer_instruction5b = "Subtask 5b: Use programmer agent to verify and finalize the validated answer from review, ensuring correctness and completeness."
    programmer_desc5b = {
        'instruction': programmer_instruction5b,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "feedback of subtask 5", "correctness of subtask 5"]
    }
    results5b = await self.programmer(
        subtask_id="subtask_6",
        programmer_desc=programmer_desc5b
    )
    agents.append(f"Programmer agent {results5b['programmer_agent'].id}, verifying final answer, thinking: {results5b['thinking'].content}; answer: {results5b['answer'].content}, executing results: {results5b['exec_result']}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results5b['thinking'].content}; answer - {results5b['answer'].content}; output - {results5b['exec_result']}")
    logs.append(results5b['subtask_desc'])
    final_answer = await self.make_final_answer(results5b['thinking'], results5b['answer'], sub_tasks, agents)
    return final_answer, logs