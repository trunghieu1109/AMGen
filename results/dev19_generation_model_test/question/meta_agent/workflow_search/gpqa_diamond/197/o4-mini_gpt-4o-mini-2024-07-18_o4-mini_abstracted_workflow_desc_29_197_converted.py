async def forward_197(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    N = self.max_sc
    # Sub-task 1
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": "Sub-task 1: Extract total cobalt concentration, initial thiocyanate concentration, and stability constants β1–β4.", "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_temp, answer_temp = await cot_agents1[i]([taskInfo], subtask_desc1["instruction"], i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking_temp.content}; answer: {answer_temp.content}")
        possible_thinkings1.append(thinking_temp)
        possible_answers1.append(answer_temp)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, "Sub-task 1: Synthesize and choose the most consistent information for quantitative inputs.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1:", sub_tasks[-1])
    # Sub-task 2
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": "Sub-task 2: Identify and classify the relevant cobalt–thiocyanato species and their formation reactions.", "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_temp, answer_temp = await cot_agents2[i]([taskInfo, thinking1, answer1], subtask_desc2["instruction"], i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking_temp.content}; answer: {answer_temp.content}")
        possible_thinkings2.append(thinking_temp)
        possible_answers2.append(answer_temp)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent classification of species.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2:", sub_tasks[-1])
    # Sub-task 3
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": "Sub-task 3: Formulate equilibrium expressions for each complex and write the mass-balance equations.", "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_temp, answer_temp = await cot_agents3[i]([taskInfo, thinking1, answer1, thinking2, answer2], subtask_desc3["instruction"], i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, thinking: {thinking_temp.content}; answer: {answer_temp.content}")
        possible_thinkings3.append(thinking_temp)
        possible_answers3.append(answer_temp)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking1, answer1, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize and choose the most consistent equilibrium expressions and mass-balance equations.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3:", sub_tasks[-1])
    # Sub-task 4
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": "Sub-task 4: Solve the system of equilibrium and mass-balance equations to obtain concentrations of all cobalt species.", "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking_temp, answer_temp = await cot_agents4[i]([taskInfo, thinking3, answer3], subtask_desc4["instruction"], i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, thinking: {thinking_temp.content}; answer: {answer_temp.content}")
        possible_thinkings4.append(thinking_temp)
        possible_answers4.append(answer_temp)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4, "Sub-task 4: Synthesize and choose the most consistent concentrations of cobalt species.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4:", sub_tasks[-1])
    # Sub-task 5 Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_desc5 = "Sub-task 5: Calculate the percentage of the dithiocyanato complex relative to total cobalt and compare against choices." + debate_instr
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinkings5 = [[] for _ in range(self.max_round)]
    all_answers5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_desc5, "context": ["user query", "thinking of subtask 4", "answer of subtask 4"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking_temp, answer_temp = await agent([taskInfo, thinking4, answer4], debate_desc5, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking4, answer4] + all_thinkings5[r-1] + all_answers5[r-1]
                thinking_temp, answer_temp = await agent(inputs, debate_desc5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking_temp.content}; answer: {answer_temp.content}")
            all_thinkings5[r].append(thinking_temp)
            all_answers5[r].append(answer_temp)
    final_decision_agent5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent5([taskInfo, thinking4, answer4] + all_thinkings5[-1] + all_answers5[-1], "Sub-task 5: Calculate final percentage and select the matching choice. Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5:", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs