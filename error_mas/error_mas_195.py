async def forward_195(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0 - Subtask 1: CoT summary of given information
    cot_inst1 = "Sub-task 1: Extract and summarize the given information: mass m, amplitude A, Hooke’s law F=-kx, potential energy U=1/2 k x^2, rest energy m c^2, relativistic kinetic energy T=(γ-1)m c^2, and list the four candidate formulas for v_max."  
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent1([taskInfo], cot_inst1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, summarizing problem, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append({"subtask_id":"stage0_subtask1","instruction":cot_inst1,"agent_collaboration":"CoT","response":{"thinking":thinking1,"answer":answer1})
    print("Step 1: ", sub_tasks[-1])

    # Stage 0 - Subtask 2: SC_CoT derive energy relation
    sc_inst2 = "Sub-task 2: Based on Sub-task 1, write total energy at turning point E_total = m c^2 + 1/2 k A^2 and at x=0 E_total = γ m c^2, derive (γ-1) m c^2 = 1/2 k A^2, and note k A^2/(2 m c^2) < 1."  
    N2 = self.max_sc
    sc_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    for agent in sc_agents2:
        t2, a2 = await agent([taskInfo, thinking1, answer1], sc_inst2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, deriving energy relation, thinking: {t2.content}; answer: {a2.content}")
        possible_thinkings2.append(t2)
        possible_answers2.append(a2)
    final_decision2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    dec2_inst = "Sub-task 2: Synthesize and choose the most consistent derivation of the energy relation from the above attempts."  
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, dec2_inst, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision2.id}, choosing energy relation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append({"subtask_id":"stage0_subtask2","instruction":sc_inst2,"agent_collaboration":"SC_CoT","response":{"thinking":thinking2,"answer":answer2})
    print("Step 2: ", sub_tasks[-1])

    # Stage 1 - Subtask 1: SC_CoT solve for γ and express v_max
    sc_inst3 = "Sub-task 3: From (γ-1) m c^2 = 1/2 k A^2, solve for γ = 1 + k A^2/(2 m c^2), then express v_max = c sqrt(1 - 1/γ^2) in terms of k A^2/(2 m c^2)."  
    N3 = self.max_sc
    sc_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    for agent in sc_agents3:
        t3, a3 = await agent([taskInfo, thinking2, answer2], sc_inst3, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, solving for gamma and v_max, thinking: {t3.content}; answer: {a3.content}")
        possible_thinkings3.append(t3)
        possible_answers3.append(a3)
    final_decision3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    dec3_inst = "Sub-task 3: Synthesize and choose the most consistent expression for v_max."  
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, dec3_inst, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision3.id}, choosing v_max expression, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append({"subtask_id":"stage1_subtask1","instruction":sc_inst3,"agent_collaboration":"SC_CoT","response":{"thinking":thinking3,"answer":answer3})
    print("Step 3: ", sub_tasks[-1])

    # Stage 2 - Subtask 1: SC_CoT compare with candidate formulas and select correct
    sc_inst4 = "Sub-task 4: Compare the derived expression for v_max with each of the four candidate formulas and identify which choice matches exactly."  
    N4 = self.max_sc
    sc_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    for agent in sc_agents4:
        t4, a4 = await agent([taskInfo, thinking3, answer3], sc_inst4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, comparing with choices, thinking: {t4.content}; answer: {a4.content}")
        possible_thinkings4.append(t4)
        possible_answers4.append(a4)
    final_decision4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    dec4_inst = "Sub-task 4: Synthesize and choose the correct choice number for v_max."  
    thinking4, answer4 = await final_decision4([taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4, dec4_inst, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision4.id}, selecting final answer, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append({"subtask_id":"stage2_subtask1","instruction":sc_inst4,"agent_collaboration":"SC_CoT","response":{"thinking":thinking4,"answer":answer4})
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs