async def forward_183(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Extract and Summarize
    instruction1 = "Sub-task 1: Extract and summarize the starting material (benzene), target molecule (2-tert-butyl-1-ethoxy-3-nitrobenzene), and reagent pool from the query."
    N1 = self.max_sc
    sc_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":instruction1,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking_i, answer_i = await sc_agents1[i]([taskInfo], instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents1[i].id}, thinking: {thinking_i.content}; answer: {answer_i.content}")
        possible_thinkings1.append(thinking_i)
        possible_answers1.append(answer_i)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1([taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent summary for the extraction.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1 Sub-task 2: Directing Effects Debate
    instruction2 = "Sub-task 2: Analyze the electronic directing effects of tert-butyl, nitro, and phenolic/ethoxy intermediates to determine positional constraints. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = []
    all_answer2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":instruction2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        round_thinks = []
        round_ans = []
        for agent in debate_agents2:
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], instruction2, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(inputs, instruction2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            round_thinks.append(thinking2)
            round_ans.append(answer2)
        all_thinking2.append(round_thinks)
        all_answer2.append(round_ans)
    final2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_fin, answer2_fin = await final2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1],
        "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_fin.content}; answer - {answer2_fin.content}")
    subtask_desc2['response'] = {"thinking":thinking2_fin,"answer":answer2_fin}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1 Sub-task 3: Strategy Debate
    instruction3 = "Sub-task 3: Determine the strategic role and timing of sulfonation (as a blocking group), reduction/diazotization (to generate phenol), and Williamson ether formation. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = []
    all_answer3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":instruction3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        round_thinks = []
        round_ans = []
        for agent in debate_agents3:
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2_fin, answer2_fin], instruction3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2_fin, answer2_fin] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(inputs, instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            round_thinks.append(thinking3)
            round_ans.append(answer3)
        all_thinking3.append(round_thinks)
        all_answer3.append(round_ans)
    final3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_fin, answer3_fin = await final3([taskInfo, thinking2_fin, answer2_fin] + all_thinking3[-1] + all_answer3[-1],
        "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_fin.content}; answer - {answer3_fin.content}")
    subtask_desc3['response'] = {"thinking":thinking3_fin,"answer":answer3_fin}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2 Sub-task 4: Assemble Sequence with SC-CoT
    instruction4 = "Sub-task 4: Assemble a step-by-step nine-reaction sequence from benzene to the target, choosing the order of Friedel–Crafts, nitration, sulfonation/desulfonation, reduction, diazonium chemistry, etherification, and work-ups."
    N4 = self.max_sc
    sc_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":instruction4,"context":["user query","thinking of subtask 2","answer of subtask 2","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"SC_CoT"}
    for i in range(N4):
        thinking4_i, answer4_i = await sc_agents4[i]([taskInfo, thinking2_fin, answer2_fin, thinking3_fin, answer3_fin], instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents4[i].id}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings4.append(thinking4_i)
        possible_answers4.append(answer4_i)
    final4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final4([taskInfo] + possible_thinkings4 + possible_answers4,
        "Sub-task 4: Synthesize and choose the most consistent step-by-step route.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Stage 2 Sub-task 5: Compare to Multiple-Choice with CoT
    instruction5 = "Sub-task 5: Compare the assembled reaction sequence to the four provided options and identify which matches the high-yield synthesis."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent5([taskInfo, thinking4, answer4], instruction5, is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":instruction5,"context":["user query","thinking of subtask 4","answer of subtask 4"],"agent_collaboration":"CoT","response":{"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs