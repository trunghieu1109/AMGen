async def forward_80(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    passage = taskInfo.content
    cot_instruction1 = "Subtask 1: Identify and extract the distance of DeAngelo Williams' touchdown run from the passage."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [passage],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting Williams' TD run distance, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_instruction2 = "Subtask 2: Identify and extract the distance of Donovan McNabb's touchdown run from the passage."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [passage],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, extracting McNabb's TD run distance, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    cot_instruction3 = "Subtask 3: Compare the extracted touchdown run distances of Williams and McNabb to determine who had the longer TD run."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [results1['answer'].content, results2['answer'].content],
        'temperature': 0.0,
        'context': ["user query", "Williams TD run", "McNabb TD run"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, comparing TD runs, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs