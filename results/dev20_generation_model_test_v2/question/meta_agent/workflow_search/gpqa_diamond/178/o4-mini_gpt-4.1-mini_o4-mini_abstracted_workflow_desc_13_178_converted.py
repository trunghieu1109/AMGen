async def forward_178(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: Classification via SC_CoT
    sc1_instruction = "Sub-task 1: Classify each given matrix W, X, Y, Z by checking unitarity, Hermiticity, positivity and trace properties to identify observables, density matrices, evolution operators."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":sc1_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], sc1_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, classifying, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr1 = "Sub-task 1: Synthesize and choose the most consistent classification of the matrices."
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, decision_instr1, is_sub_task=True)
    agents.append(f"Final Decision Agent1, synthesizing classification, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Sub-task 2: Compute e^X and unitarity via SC_CoT
    sc2_instruction = "Sub-task 2: Compute the exponential e^X and determine whether e^X is unitary by testing if there exists any vector whose norm changes under e^X."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":sc2_instruction,"context":["user query", thinking1, answer1],"agent_collaboration":"SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], sc2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, computing e^X, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr2 = "Sub-task 2: Synthesize and choose the most consistent results for the unitarity of e^X."
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, decision_instr2, is_sub_task=True)
    agents.append(f"Final Decision Agent2, synthesizing e^X unitarity, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Sub-task 4: Debate to choose correct statement
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction4 = "Sub-task 4: Evaluate each of the four provided statements in light of the classification and computations, and select the correct one." + debate_instr
    debate_agents4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max4)]
    all_answer4 = [[] for _ in range(N_max4)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":debate_instruction4,"context":["user query", thinking2, answer2],"agent_collaboration":"Debate"}
    for r in range(N_max4):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                inputs = [taskInfo, thinking2, answer2]
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking4[r-1] + all_answer4[r-1]
            thinking4_i, answer4_i = await agent(inputs, debate_instruction4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking2, answer2] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: Evaluate each of the four provided statements in light of the classification and computations, and select the correct one." + final_instr4, is_sub_task=True)
    agents.append(f"Final Decision agent4, selecting correct statement, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 3: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs