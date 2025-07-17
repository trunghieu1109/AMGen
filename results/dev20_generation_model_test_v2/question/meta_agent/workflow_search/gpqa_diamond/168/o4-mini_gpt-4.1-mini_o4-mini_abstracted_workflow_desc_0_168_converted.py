async def forward_168(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    N_sc = self.max_sc
    # Stage 1 Sub-task 1: SC-CoT
    sc1 = "Sub-task 1: Analyze the original decay 2A→2B+2E+2V and explain why the E energy spectrum is continuous with endpoint Q"
    cot_agents1 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N_sc)]
    possible_thinks1 = []
    possible_ans1 = []
    subtask_desc1 = {"subtask_id":"stage1_subtask1","instruction":sc1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i,agent in enumerate(cot_agents1):
        thinking1_i, answer1_i = await agent([taskInfo], sc1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, analyzing original decay, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinks1.append(thinking1_i)
        possible_ans1.append(answer1_i)
    final1 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking1, answer1 = await final1([taskInfo] + possible_thinks1 + possible_ans1, "Sub-task 1: Synthesize and choose the most consistent explanation for original decay", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response']={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 1 Sub-task 2: SC-CoT
    sc2 = "Sub-task 2: Analyze the variant decay 2A→2B+2E+M and note that M is massless and one species replaces two V’s"
    cot_agents2 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N_sc)]
    possible_thinks2 = []
    possible_ans2 = []
    subtask_desc2 = {"subtask_id":"stage1_subtask2","instruction":sc2,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i,agent in enumerate(cot_agents2):
        thinking2_i, answer2_i = await agent([taskInfo], sc2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, analyzing variant decay, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinks2.append(thinking2_i)
        possible_ans2.append(answer2_i)
    final2 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking2, answer2 = await final2([taskInfo] + possible_thinks2 + possible_ans2, "Sub-task 2: Synthesize and choose the most consistent explanation for variant decay", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response']={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 2 Sub-task 1: Debate
    debate3 = "Sub-task 3: Assess the kinematic impact of replacing two V particles by one massless M on the phase-space available to the two E particles. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"],"Debate Agent",model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    rounds = self.max_round
    all_thk3 = [[] for _ in range(rounds)]
    all_ans3 = [[] for _ in range(rounds)]
    subtask_desc3 = {"subtask_id":"stage2_subtask1","instruction":debate3,"context":["user query","output subtask1","output subtask2"],"agent_collaboration":"Debate"}
    for r in range(rounds):
        for agent in debate_agents:
            if r==0:
                thinking3_i, answer3_i = await agent([taskInfo,thinking1,answer1,thinking2,answer2], debate3, r, is_sub_task=True)
            else:
                inputs = [taskInfo,thinking1,answer1,thinking2,answer2] + all_thk3[r-1] + all_ans3[r-1]
                thinking3_i, answer3_i = await agent(inputs, debate3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thk3[r].append(thinking3_i)
            all_ans3[r].append(answer3_i)
    final3 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking3, answer3 = await final3([taskInfo,thinking1,answer1,thinking2,answer2] + all_thk3[-1] + all_ans3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response']={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Stage 3 Sub-task 1: SC-CoT
    sc4 = "Sub-task 4: Derive whether the E-particle energy spectrum remains continuous or becomes discrete, and describe how its shape is altered."
    cot_agents4 = [LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N_sc)]
    possible_thk4 = []
    possible_ans4 = []
    subtask_desc4 = {"subtask_id":"stage3_subtask1","instruction":sc4,"context":["user query","output subtask3"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents4:
        thinking4_i, answer4_i = await agent([taskInfo,thinking3,answer3], sc4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, analyzing spectrum continuity, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thk4.append(thinking4_i)
        possible_ans4.append(answer4_i)
    final4 = LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking4, answer4 = await final4([taskInfo,thinking3,answer3] + possible_thk4 + possible_ans4, "Sub-task 4: Synthesize and choose the most consistent determination for spectrum continuity and shape change", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response']={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Stage 4 Sub-task 1: Reflexion
    reflect5 = "Sub-task 5: Combine the phase-space analysis with energy–momentum conservation to determine the endpoint shift and map to the provided choices. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot5 = LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    critic = LLMAgentBase(["feedback","correct"],"Critic Agent",model=self.node_model,temperature=0.0)
    cot_inputs5 = [taskInfo,thinking1,answer1,thinking2,answer2,thinking3,answer3,thinking4,answer4]
    subtask_desc5 = {"subtask_id":"stage4_subtask1","instruction":reflect5,"context":["user query","outputs of subtask1-4"],"agent_collaboration":"Reflexion"}
    thinking5, answer5 = await cot5(cot_inputs5, reflect5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        feedback, correct = await critic([taskInfo,thinking5,answer5],"Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content=="True":
            break
        cot_inputs5.extend([thinking5,answer5,feedback])
        thinking5, answer5 = await cot5(cot_inputs5, reflect5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response']={"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs