async def forward_180(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Identify all solar neutrino sources in the 700–800 keV and 800–900 keV bands (Debate)
    debate_instr = (
        "Sub-task 1: Identify solar neutrino sources contributing to the 700–800 keV and 800–900 keV energy bands, distinguishing line versus continuum components." 
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                     for role in self.debate_role]
    all_thinking1 = [[] for _ in range(self.max_round)]
    all_answer1 = [[] for _ in range(self.max_round)]
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":debate_instr,"context":["user query"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr, r, is_sub_task=True)
            else:
                inputs = [taskInfo] + all_thinking1[r-1] + all_answer1[r-1]
                thinking, answer = await agent(inputs, debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking1[r].append(thinking)
            all_answer1[r].append(answer)
    final_decision1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision1(
        [taskInfo] + all_thinking1[-1] + all_answer1[-1],
        "Sub-task 1: Identify solar neutrino sources contributing to these bands. Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1:", sub_tasks[-1])

    # Sub-task 2: Compute numerical flux contributions before stopping pp-III (SC_CoT)
    cot_sc_instruction = (
        "Sub-task 2: Using standard solar neutrino fluxes, compute the numerical contributions of each identified source to the flux in the 700–800 keV and 800–900 keV bands before pp-III is stopped."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                  for _ in range(N)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query", thinking1.content, answer1.content],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decider2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decider2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent numerical flux contributions for each band.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2:", sub_tasks[-1])

    # Sub-task 3: Remove pp-III continuum contributions (SC_CoT)
    cot_sc_instruction3 = (
        "Sub-task 3: Model the stoppage of the pp-III (⁸B) branch by removing its continuum contributions from the previously computed band fluxes."
    )
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                   for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction3,"context":["user query", thinking2.content, answer2.content],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents3:
        thinking, answer = await agent([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings3.append(thinking)
        possible_answers3.append(answer)
    final_decider3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decider3(
        [taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Synthesize and choose the updated band fluxes after pp-III stoppage.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3:", sub_tasks[-1])

    # Sub-task 4: Calculate post-stoppage flux ratio (CoT)
    cot_instruction4 = (
        "Sub-task 4: Calculate the ratio Flux(700–800 keV) / Flux(800–900 keV) using the updated band fluxes after pp-III stoppage."
    )
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_instruction4,"context":["user query", thinking3.content, answer3.content],"agent_collaboration":"CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4:", sub_tasks[-1])

    # Sub-task 5: Compare to provided options and select closest match (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = (
        "Sub-task 5: Compare the computed ratio to the choices [0.1, 10, 1, 0.01] and select the closest match." + reflect_inst
    )
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_reflect_instruction,"context":["user query", thinking4.content, answer4.content],"agent_collaboration":"Reflexion"}
    cot_inputs5 = [taskInfo, thinking4, answer4]
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(self.max_round):
        critic_inst = (
            "Please review the answer above and criticize where it might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'."
        )
        feedback, correct = await critic_agent5([taskInfo, thinking5, answer5], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True":
            break
        cot_inputs5 += [thinking5, answer5, feedback]
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refining: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs