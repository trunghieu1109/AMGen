async def forward_26(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Understand the problem scenario: Alice chooses a set A and Bob lists all finite nonempty sets B whose maximum is in A. Describe the scenario and what is being asked."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, scenario understanding, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    subtask_desc1["response"]={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    cot_sc_instruction = "Sub-task 2: For a fixed positive integer a, derive the count of finite nonempty sets B with max(B)=a, noting that B can be any subset of {1,...,a} including a."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for i, agent in enumerate(cot_agents):
        thinking2, answer2 = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, derive count for max(B)=a, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content]=thinking2
        answermapping[answer2.content]=answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    subtask_desc2["response"]={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    cot_reflect_instruction = "Sub-task 3: Combine the result from Sub-task 2 to express the total number of sets as the sum over a in A of 2^(a-1)."
    cot_agent = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"],"Critic Agent",model=self.node_model,temperature=0.0)
    N_max = self.max_round
    cot_inputs = [taskInfo, thinking1, answer1, thinking2, answer2]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_reflect_instruction,"context":["user query","thinking of subtask 1","answer of subtask 1","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Reflexion"}
    thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, combine counts, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Please review the expression for total number of sets and point out errors.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, refine combination, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    subtask_desc3["response"]={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    cot_instruction = "Sub-task 4: Set up the equation sum_{a in A} 2^(a-1) = 2024 based on Bob's list size."
    cot_agent = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_instruction,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"CoT"}
    thinking4, answer4 = await cot_agent([taskInfo, thinking3, answer3], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, set up equation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    subtask_desc4["response"]={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    debate_instruction = "Sub-task 5: Solve sum_{a in A} 2^(a-1)=2024 by finding binary representation of 2024 and identifying corresponding a values."
    debate_agents = [LLMAgentBase(["thinking","answer"],"Debate Agent",model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking5 = [[] for _ in range(N_max)]
    all_answer5 = [[] for _ in range(N_max)]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":debate_instruction,"context":["user query","thinking of subtask 4","answer of subtask 4"],"agent_collaboration":"Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction, r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, solve binary representation, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking5, answer5 = await final_decision_agent([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on a values.", is_sub_task=True)
    agents.append(f"Final Decision Agent, deciding a values, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    subtask_desc5["response"]={"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    cot_instruction = "Sub-task 6: Compute the sum of elements of A from the values identified in Sub-task 5."
    cot_agent = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot_instruction,"context":["user query","thinking of subtask 5","answer of subtask 5"],"agent_collaboration":"CoT"}
    thinking6, answer6 = await cot_agent([taskInfo, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, compute sum of A, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    subtask_desc6["response"]={"thinking":thinking6,"answer":answer6}
    logs.append(subtask_desc6)
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
