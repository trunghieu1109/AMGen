async def forward_5(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction1 = "Subtask 1: Identify the Rome Protocols and the three Prime Ministers who signed them with detailed reasoning and context from the taskInfo"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying Rome Protocols and signatories, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_instruction2 = "Subtask 2: Determine which of the three Prime Ministers identified in Subtask 1 was assassinated, using detailed reasoning and context from previous outputs"
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
    agents.append(f"CoT agent {results2['cot_agent'].id}, determining assassinated Prime Minister, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    cot_instruction3 = "Subtask 3: Identify the event or context related to the assassination of the Prime Minister determined in Subtask 2, with detailed reasoning"
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, identifying assassination event context, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    aggregate_instruction4 = "Subtask 4: Consolidate the candidate outputs from Subtasks 1, 2, and 3 into a coherent and consistent answer about the assassination event and related context"
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo, results1['answer'], results2['answer'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "answers from subtasks 1, 2, 3"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating candidate outputs, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    review_instruction5 = "Subtask 5: Validate the consolidated answer from Subtask 4 for accuracy, completeness, and correctness regarding the assassination context"
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
    agents.append(f"Review agent {results5['review_agent'].id}, validating consolidated answer, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    programmer_instruction6 = "Subtask 6: Generate a final refined answer based on the review feedback and consolidated answer to ensure clarity and completeness"
    programmer_desc6 = {
        'instruction': programmer_instruction6,
        'input': [taskInfo, results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4", "feedback of subtask 5", "correct of subtask 5"]
    }
    results6 = await self.programmer(
        subtask_id="subtask_6",
        programmer_desc=programmer_desc6
    )
    agents.append(f"Programmer agent {results6['programmer_agent'].id}, refining final answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    cot_instruction7 = "Subtask 7: Final check and reasoning to ensure the refined answer fully addresses the query about the Rome Protocols and the assassination context"
    cot_agent_desc7 = {
        'instruction': cot_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7 = await self.cot(
        subtask_id="subtask_7",
        cot_agent_desc=cot_agent_desc7
    )
    agents.append(f"CoT agent {results7['cot_agent'].id}, final reasoning check, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
