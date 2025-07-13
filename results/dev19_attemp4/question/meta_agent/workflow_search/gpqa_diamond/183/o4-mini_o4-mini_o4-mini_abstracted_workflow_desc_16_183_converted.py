async def forward_183(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: SC_CoT
    sc_instruction1 = "Sub-task 1: Analyze the directing effects of tert-butyl, ethoxy, and nitro substituents on benzene and determine the need for a sulfonation/desulfonation blocking strategy."
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask1_desc = {"subtask_id":"subtask_1","instruction":sc_instruction1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking_i, answer_i = await cot_agents1[i]([taskInfo], sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, analyzing directing effects, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr1 = "Given all the above thinking and answers, find the most consistent analysis of directing effects and blocking strategy."
    thinking1, answer1 = await final_agent1([taskInfo] + possible_thinkings1 + possible_answers1, decision_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask1_desc['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask1_desc)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: CoT
    cot_instruction2 = "Sub-task 2: Propose a retrosynthetic sequence for installing tert-butyl, nitro, and ethoxy groups in positions 2, 3, and 1 respectively—outlining the order of Friedel–Crafts alkylation, sulfonation/nitration, desulfonation, nitro reduction/diazotization/hydrolysis, and Williamson ether synthesis."
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask2_desc = {"subtask_id":"subtask_2","instruction":cot_instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"CoT"}
    thinking2, answer2 = await cot_agent2([taskInfo, thinking1, answer1], cot_instruction2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent2.id}, propose retrosynthetic sequence, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask2_desc['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask2_desc)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Debate
    debate_instruction3 = "Sub-task 3: Map the retrosynthetic plan step-by-step onto each of the four given choice sequences, labeling reagents i–ix and checking for correct order and compatibility. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N3 = self.max_round
    all_thinking3 = [[] for _ in range(N3)]
    all_answer3 = [[] for _ in range(N3)]
    subtask3_desc = {"subtask_id":"subtask_3","instruction":debate_instruction3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Debate"}
    for r in range(N3):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(inputs3, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, mapping sequences, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all the above thinking and answers, reason over them carefully and provide a final answer for the mapping of choice sequences."
    thinking3, answer3 = await final_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask3_desc['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask3_desc)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: SC_CoT
    sc_instruction4 = "Sub-task 4: Evaluate each mapped sequence for proper regioselectivity, blocking‐group deployment, and functional-group interconversions, and select the sequence that affords the target in high yield."
    N4 = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask4_desc = {"subtask_id":"subtask_4","instruction":sc_instruction4,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"SC_CoT"}
    for i in range(N4):
        thinking4_i, answer4_i = await cot_agents4[i]([taskInfo, thinking3, answer3], sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, evaluate sequences, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_instr4 = "Given all the above thinking and answers, choose the best sequence that leads to high-yield synthesis."
    thinking4, answer4 = await final_agent4([taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4, decision_instr4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask4_desc['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask4_desc)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Reflexion
    reflect_inst5 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction5 = "Sub-task 5: Summarize the chosen reaction sequence and provide a concise justification based on directing effects, blocking strategy, and step-wise transformations." + reflect_inst5
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask5_desc = {"subtask_id":"subtask_5","instruction":cot_reflect_instruction5,"context":["user query","thinking of subtask 1","answer of subtask 1","thinking of subtask 2","answer of subtask 2","thinking of subtask 3","answer of subtask 3","thinking of subtask 4","answer of subtask 4"],"agent_collaboration":"Reflexion"}
    N5 = self.max_round
    thinking5, answer5 = await cot_agent5([taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3, thinking4, answer4], cot_reflect_instruction5, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, initial summary and justification, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N5):
        critic_inst5 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], critic_inst5, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, critique, thinking: {feedback5.content}; answer: {correct5.content}")
        if correct5.content == "True":
            break
        thinking5, answer5 = await cot_agent5([taskInfo, thinking5, answer5, feedback5], cot_reflect_instruction5, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refined summary, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask5_desc['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask5_desc)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs