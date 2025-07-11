async def forward_194(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Compute the semi-major axis a1 of the known planet from its 3 day period using Kepler’s third law and estimated stellar mass from R*=1.5Rsun."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, computing a1, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"]={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction = "Sub-task 2: Derive the orbital inclination i using impact parameter b1=0.2 and result from Sub-task 1."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","thinking1","answer1"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking2, answer2 = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, deriving i, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content] = thinking2
        answermapping[answer2.content] = answer2
    answer2_content = Counter(possible_answers).most_common(1)[0][0]
    thinking2 = thinkingmapping[answer2_content]
    answer2 = answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"]={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    sc_instruction3 = "Sub-task 3: Calculate maximum semi-major axis a2_max for second planet with Rp2=2.5Re and derived inclination i."
    M = self.max_sc
    sc_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(M)]
    possible3 = []
    thinkingmap3 = {}
    answermap3 = {}
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":sc_instruction3,"context":["user query","thinking2","answer2"],"agent_collaboration":"SC_CoT"}
    for j in range(M):
        thinking3, answer3 = await sc_agents3[j]([taskInfo, thinking2, answer2], sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents3[j].id}, calculating a2_max, thinking: {thinking3.content}; answer: {answer3.content}")
        possible3.append(answer3.content)
        thinkingmap3[answer3.content] = thinking3
        answermap3[answer3.content] = answer3
    answer3_content = Counter(possible3).most_common(1)[0][0]
    thinking3 = thinkingmap3[answer3_content]
    answer3 = answermap3[answer3_content]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"]={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    debate_instruction = "Sub-task 4: Convert a2_max into P2_max using Kepler’s third law, compare to choices and select the correct orbital period."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    R = self.max_round
    all_thinking4 = [[] for _ in range(R)]
    all_answer4 = [[] for _ in range(R)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":debate_instruction,"context":["user query","thinking3","answer3"],"agent_collaboration":"Debate"}
    for r in range(R):
        for k, agent in enumerate(debate_agents):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing P2_max, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_agent = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_agent([taskInfo] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Make final decision on P2_max.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_agent.id}, selecting final P2_max, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"]={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs