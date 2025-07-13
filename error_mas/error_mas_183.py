async def forward_183(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction = "Sub-task 1: Extract and summarize the starting material, target substitution pattern (2-tert-butyl, 1-ethoxy, 3-nitro), and list all available reagents/conditions."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking1_i, answer1_i = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings.append(thinking1_i)
        possible_answers.append(answer1_i)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize and choose the most consistent extraction and summary.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    debate_instruction_2 = "Sub-task 2: Classify each reagent/condition by its role (e.g., Friedel–Crafts alkylation, nitration, sulfonation, reduction, diazotization, ether formation). Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":debate_instruction_2,"context":["user query",thinking1,answer1],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents_2:
            if r == 0:
                thinking2_r, answer2_r = await agent([taskInfo, thinking1, answer1], debate_instruction_2, r, is_sub_task=True)
            else:
                thinking2_r, answer2_r = await agent([taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1], debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2_r.content}; answer: {answer2_r.content}")
            all_thinking2[r].append(thinking2_r)
            all_answer2[r].append(answer2_r)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)\n    print("Step 2: ", sub_tasks[-1])
    debate_instruction_3 = "Sub-task 3: Analyze directing effects and regiochemical requirements to determine the order of install/removal steps needed to place tert-butyl, nitro, and ethoxy at the correct positions. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instruction_3,"context":["user query",thinking2,answer2],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents_3:
            if r == 0:
                thinking3_r, answer3_r = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                thinking3_r, answer3_r = await agent([taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1], debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_r.content}; answer: {answer3_r.content}")
            all_thinking3[r].append(thinking3_r)
            all_answer3[r].append(answer3_r)
    final_decision_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_instruction_4 = "Sub-task 4: Generate candidate multi-step routes matching the required directing-group manipulations (using sulfonation as a block, reduction/diazotization for phenol, Williamson etherification) and map them onto the provided sequences."
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction_4,"context":["user query",thinking3,answer3],"agent_collaboration":"SC_CoT"}
    for i in range(N4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_decision_agent_4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4, "Sub-task 4: Synthesize and choose the most consistent multi-step route.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    debate_instruction_5 = "Sub-task 5: Evaluate the regiochemical correctness and practicality (high-yield assumptions) of each mapped sequence and select the best one. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking5 = [[] for _ in range(self.max_round)]
    all_answer5 = [[] for _ in range(self.max_round)]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":debate_instruction_5,"context":["user query",thinking4,answer4],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents_5:
            if r == 0:
                thinking5_r, answer5_r = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                thinking5_r, answer5_r = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5_r.content}; answer: {answer5_r.content}")
            all_thinking5[r].append(thinking5_r)
            all_answer5[r].append(answer5_r)
    final_decision_agent_5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs