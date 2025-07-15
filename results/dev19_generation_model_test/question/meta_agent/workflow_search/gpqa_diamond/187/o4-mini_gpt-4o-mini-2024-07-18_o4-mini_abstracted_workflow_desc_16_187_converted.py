async def forward_187(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: SC_CoT
    cot_sc_instruction1 = "Sub-task 1: Extract and interpret the lattice parameter a, the lattice angles alpha, and the Miller indices from the query."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents1:
        thinking1_i, answer1_i = await agent([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Given all the above thinking and answers, determine the consistent interpretation of the crystallographic inputs."
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Sub-task 2: SC_CoT
    cot_sc_instruction2 = "Sub-task 2: Construct the rhombohedral metric tensor g_ij using a and alpha, and compute its determinant."
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all the above thinking and answers, choose the most consistent metric tensor and determinant."
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Sub-task 3: Debate
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction3 = "Sub-task 3: Derive the interplanar spacing formula for a triclinic cell and simplify it for the (111) plane in the rhombohedral lattice." + debate_instr
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N3 = self.max_round
    all_thinking3 = [[] for _ in range(N3)]
    all_answer3 = [[] for _ in range(N3)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instruction3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Debate"}
    for r in range(N3):
        for agent in debate_agents3:
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(inputs3, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Sub-task 4: SC_CoT
    cot_sc_instruction4 = "Sub-task 4: Plug in a=10 Angstrom and alpha=30 degrees into the simplified d_111 expression, compute the numeric result, and match to the provided options."
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction4,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents4:
        thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, select the most consistent numerical result for d_111."
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4, final_instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs