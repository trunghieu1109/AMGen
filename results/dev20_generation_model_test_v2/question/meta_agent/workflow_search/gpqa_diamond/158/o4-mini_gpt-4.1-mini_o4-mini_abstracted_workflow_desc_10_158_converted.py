async def forward_158(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_sc_instruction = "Sub-task 0: Determine the quasar’s redshift by mapping the observed 790 nm flux drop to the rest-frame Lyman-α line (121.6 nm) and computing z = λ_obs/λ_rest – 1."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc0 = {"subtask_id":"subtask_0","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking0, answer0 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, determine redshift, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings.append(thinking0)
        possible_answers.append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr0 = "Sub-task 0: Synthesize and choose the most consistent answer for the quasar redshift. Given all the above thinking and answers, find the most consistent and correct solution for the quasar redshift."
    thinking0, answer0 = await final_decision_agent_0([taskInfo] + possible_thinkings + possible_answers, synth_instr0, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_0.id}, synthesizing redshift, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking":thinking0,"answer":answer0}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])
    cot_sc_instruction1 = "Sub-task 1: Compute the comoving distance D_C for the derived redshift using D_C = c ∫₀ᶻ dz'/H(z') with H(z') = H₀√[Ωₘ(1+z')³ + Ω_Λ]."
    cot_agents1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction1,"context":["user query","thinking of subtask 0","answer of subtask 0"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking1, answer1 = await cot_agents1[i]([taskInfo, thinking0, answer0], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, compute comoving distance, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings1.append(thinking1)
        possible_answers1.append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr1 = "Sub-task 1: Synthesize and choose the most consistent answer for the comoving distance. Given all the above thinking and answers, find the most consistent and correct solution for the comoving distance."
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings1 + possible_answers1, synth_instr1, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_1.id}, synthesizing comoving distance, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])
    debate_instr = "Sub-task 2: Compare the calculated comoving distance to the provided choices (6, 7, 8, 9 Gpc) and select the nearest value. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking2 = [[] for _ in range(N_max)]
    all_answer2 = [[] for _ in range(N_max)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":debate_instr,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"Debate"}
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1, answer1], debate_instr, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2, answer2 = await agent(inputs, debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2:" + final_instr2, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2.id}, calculating final choice, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs